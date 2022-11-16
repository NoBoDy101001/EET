import copy
import math
import time
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module, ModuleList

from typing import Any, Dict, List, Optional, Tuple, cast

from fairscale.nn import MOELayer, Top2Gate

from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name
from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import LayerNorm as eet_layernorm

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


class EETExpert():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, bias=True, name="out_cache", device=0):
        self.intermediate_weights = torch.t(model_dict['experts.' + str(layer_id) + '.ffn.intermediate.weight']).contiguous().type(data_type).to(device)
        self.intermediate_bias = model_dict['experts.' + str(layer_id) + '.ffn.intermediate.bias'].type(data_type).to(device) if bias else torch.empty(0)
        self.output_weights = torch.t(model_dict['experts.' + str(layer_id) + '.ffn.output.weight']).contiguous().type(data_type).to(device)
        self.output_bias = model_dict['experts.' + str(layer_id) + '.ffn.output.bias'].type(data_type).to(device) if bias else torch.empty(0)
        self.layernorm_weights = torch.empty(0)
        self.layernorm_bias = torch.empty(0)

        self.ffn = eet_ffn(config, self.intermediate_weights, self.intermediate_bias, self.output_weights, self.output_bias, self.layernorm_weights, self.layernorm_bias, name)

    def __call__(
        self,
        hidden_states,
        pre_layernorm=False,
        add_residual=False
    ):
        return self.ffn.forward(hidden_states, pre_layernorm, add_residual)

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, bias=True, name="out_cache", device=0):
        expert = EETExpert(config, model_dict, layer_id, data_type=data_type, bias=bias, name=name, device=device)
        return expert

class EETMOELayer():
    def __init__(self, gate, experts, group=None):
        super().__init__()
        self.gate = gate
        if type(experts) == List:
            self.experts = cast(List, experts)
        else:
            self.experts = list(experts)
        self.group = group if group is not None else dist.group.WORLD
        # for expert in self.experts:
        #     for p in experts.parameters():
        #         p.expert = True  # type: ignore
        self.world_size = dist.get_world_size(self.group)
        self.num_local_experts = len(self.experts)

    def __call__(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        assert input[0].shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[2]
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input[0].reshape(-1, d_model)
        self.l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input)
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.float(), reshaped_input)
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        # Re-shape after all-to-all: ecm -> gecm
        # fix-bug: eet input should be in contiguous mem cache
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model)
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            chunk = chunk.contiguous()
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        expert_output = _AllToAll.apply(self.group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, d_model)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output)
        return combined_output.reshape(input[0].shape)
    
    @staticmethod
    def from_torch(moe_layer_dict, data_type=torch.float32, bias=True, device=0):
        """EET moe layer."""
        torch.set_grad_enabled(False)
        num_local_experts = 4
        batch_size = 4
        model_dim = 8
        activation_fn = "relu"
        device = "cpu" if device < 0 else f"cuda:{device}"
        world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
        num_experts = num_local_experts * world_size
        # batch_size *= world_size
        print("batch size: ", batch_size)

        experts_dict = {}
        model_name = "moe"
        
        for k, v in moe_layer_dict.items():
            if 'experts.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('experts.'):]
                experts_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(experts_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}        
        
        config = meta_desc(batch_size, 12, model_dim, 6,
                           512, 512, data_type, device, False,
                           activation_fn)
        experts = []
        # eet ffn output mem need to be set 
        for i in range(num_local_experts):
            experts.append(EETExpert.from_torch(config, layer_model_dict['experts.' + str(i)], i, data_type=data_type, bias=bias, name="expert_out" + str(i), device=device))
        gate = Top2Gate(model_dim, num_experts)
        gate.wg.weight = torch.nn.Parameter(moe_layer_dict['gate.wg.weight'])
        gate = gate.to(device)
        return EETMOELayer(gate=gate, experts=experts)