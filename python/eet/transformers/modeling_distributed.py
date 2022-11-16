#
# Created by djz on 2022/10/27.
#
"""EET distributed model. """
# TODO
import copy
import math
import copy
import time
import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
import torch.distributed as dist
from torch import Tensor

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from transformers import  (
    BertModel,
    GPT2Model,
    T5Model,
)

from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import Embedding as eet_embedding

__all__ = ['EETPipeBertModel', 'EETPipeGPTModel']


class EETBertEmbedding():
    def __init__(self, config, embedding_dict, data_type=torch.float32, name='emb_cache', device=0):
        self.if_layernorm = True
        self.embedding_weights = embedding_dict['embeddings.word_embeddings.weight'].type(data_type).to(device)
        self.position_weights = embedding_dict['embeddings.position_embeddings.weight'].type(data_type).to(device)
        self.token_type_weights = embedding_dict['embeddings.token_type_embeddings.weight'].type(data_type).to(device)
        self.Layernorm_weights = embedding_dict['embeddings.LayerNorm.weight'].type(data_type).to(device)
        self.Layernorm_bias = embedding_dict['embeddings.LayerNorm.bias'].type(data_type).to(device)
        self.embedding = eet_embedding(config,self.embedding_weights,self.position_weights,self.token_type_weights,self.Layernorm_weights,self.Layernorm_bias, name)

    def __call__(self,
                input_ids,
                position_ids,
                token_type_ids):
        return self.embedding.forward_transformers(input_ids,position_ids,token_type_ids,self.if_layernorm)
    
    @staticmethod
    def from_torch(config, embedding_dict, data_type=torch.float32, name='emb_cache', device=0):
        embedding = EETBertEmbedding(config, embedding_dict, data_type=data_type, name=name, device=device)
        return embedding

class EETGPT2Embedding():
    def __init__(self,meta_des,embedding_dict,data_type = torch.float32, name='emb_cache', device=0):
        self.embedding_weights = embedding_dict['wte.weight'].type(data_type).to(device)
        self.position_weights = embedding_dict['wpe.weight'].type(data_type).to(device)
        # not use token_type
        self.token_type_ids =  torch.empty(0).long()
        self.token_type_weights = torch.empty(0)
        self.if_layernorm = False
        # not use layernorm
        self.Layernorm_weights = torch.empty(0)
        self.Layernorm_bias = torch.empty(0)
        self.embedding = eet_embedding(meta_des,self.embedding_weights,self.position_weights,self.token_type_weights,self.Layernorm_weights,self.Layernorm_bias, name)
    def __call__(self,
                input_ids,
                position_ids,
                token_type_ids):
        if_layernorm = False
        if token_type_ids is None:
            token_type_ids = self.token_type_ids
        return self.embedding.forward_transformers(input_ids, position_ids, token_type_ids, if_layernorm)

    @staticmethod
    def from_torch(meta_des, embedding_dict, data_type=torch.float32, device=0):
        feedforward = EETGPT2Embedding(meta_des, embedding_dict, data_type=data_type, device=device)
        return feedforward

class EETPipeBertModel():
    def __init__(self, config, partitions, balances, devices):
        self.partitions = partitions
        self.balances = balances
        self.devices = devices
        self.pre_padding_len = torch.empty(0).long()
        self.position_ids = torch.arange(0, config.max_position_embeddings).reshape(1, config.max_position_embeddings).to(devices[0])

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        '''
        attention_mask:attention_padding_mask(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input.)
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        '''
        first_device = self.devices[0]
        input_shape = input_ids.size()
        position_ids = self.position_ids[:, :input_shape[1]]

        
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=first_device) if token_type_ids is None else token_type_ids.to(first_device)
        
        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len =  torch.sum(1 - attention_mask,1).long().to(first_device)

        hidden_states = input_ids
        layer_id = 0
        for i in range(len(self.balances)):
            torch.cuda.set_device(self.devices[i])
            hidden_states = hidden_states.to(self.devices[i], non_blocking=True)
            pre_padding_len = pre_padding_len.to(self.devices[i], non_blocking=True)
            for _ in range(self.balances[i]):
                layer = self.partitions[layer_id]
                # embedding
                if layer_id == 0:
                    hidden_states = layer(hidden_states, position_ids, token_type_ids)
                else:
                    hidden_states = layer(
                        hidden_states,
                        pre_padding_len=pre_padding_len,
                        normalize_before=False,
                    )
                layer_id += 1
        return hidden_states


    @staticmethod
    def from_pretrained(model_id_or_path: str,max_batch, data_type, balances=None):
        """EET pipeline parallel bert model."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = BertModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        model_name = "bert"

        for k, v in torch_model.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        configs = []
        devices = []
        partitions = []
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        layer_num = cfg.num_hidden_layers + 1

        # init eet config
        if isinstance(balances, List):
            assert sum(balances) == layer_num, "sum of balances is not equal to module layer num"
        else:
            print("input balance is not a list, so run with default device")
            balances = [cfg.num_hidden_layers + 1]
        for i in range(len(balances)):
            device = f"cuda:{i}"
            config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers,
                               cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False,
                               activation_fn)
            configs.append(config)
            devices.append(device)

        # TODO partitions
        layer_id = 0
        for i in range(len(balances)):
            tmp = []
            for _ in range(balances[i]):
                # embedding
                if layer_id == 0:
                    partitions.append(EETBertEmbedding.from_torch(configs[0], embedding_dict, data_type, device=devices[0]))
                # encoder layer
                else:
                    partitions.append(EETEncoderLayer.from_torch(configs[i], layer_model_dict['layer.' + str(layer_id-1)], layer_id-1, data_type=data_type, bias=True, device=devices[i]))
                layer_id += 1
        eet_model = EETPipeBertModel(cfg, partitions, balances, devices)
        return eet_model

    def from_torch(torch_model, max_batch, data_type, balances=None):
        """EET pipeline parallel bert model."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        cfg = torch_model.config
        model_name = "bert"

        for k, v in torch_model.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        configs = []
        devices = []
        partitions = []
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        layer_num = cfg.num_hidden_layers + 1

        # init eet config
        if isinstance(balances, List):
            assert sum(balances) == layer_num, "sum of balances is not equal to module layer num"
        else:
            print("input balance is not a list, so run with default device")
            balances = [cfg.num_hidden_layers + 1]
        for i in range(len(balances)):
            device = f"cuda:{i}"
            config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers,
                               cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False,
                               activation_fn)
            configs.append(config)
            devices.append(device)

        # TODO partitions
        layer_id = 0
        for i in range(len(balances)):
            for _ in range(balances[i]):
                # embedding
                if layer_id == 0:
                    partitions.append(EETBertEmbedding.from_torch(configs[0], embedding_dict, data_type, device=devices[0]))
                # encoder layer
                else:
                    partitions.append(EETEncoderLayer.from_torch(configs[i], layer_model_dict['layer.' + str(layer_id-1)], layer_id-1, data_type=data_type, bias=True, device=devices[i]))
                layer_id += 1
        eet_model = EETPipeBertModel(cfg, partitions, balances, devices)
        return eet_model


class EETPipeGPTModel():
    def __init__(self, config, partitions, layer_norm, balances, devices):
        self.partitions = partitions
        self.layer_norm = layer_norm
        self.balances = balances
        self.devices = devices
        self.position_ids = torch.arange(0, config.n_positions).reshape(1, config.n_positions).to(devices[0])
        self.self_attn_padding_mask = torch.empty(0).long()
        self.encoder_attention_mask = torch.empty(0)
        self.reorder_state = torch.empty(0).long()
        self.current_len = 0

    def __call__(
        self,
        input_ids,
        encoder_outputs=None,
        first_pass=True,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        reorder_state=None,
    ):
        '''
        attention_mask:attention_padding_mask(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input.)
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        '''
        first_device = self.devices[0]
        input_shape = input_ids.size()

        # Attention mask.
        if attention_mask is None:
            pre_padding_len = self.self_attn_padding_mask
        else:
            pre_padding_len =  torch.sum(1 - attention_mask,1).long().to(first_device)

        if position_ids is None:
            if first_pass is True:
                self.current_len = input_shape[1]
                position_ids = self.position_ids[:, :self.current_len]
            else:
                self.current_len = self.current_len + 1
                position_ids = self.position_ids[:, self.current_len-1:self.current_len]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1]).to(first_device)

        if reorder_state is not None:
            self.reorder_state = reorder_state.long().to(first_device)

        hidden_states = input_ids
        layer_id = 0
        for i in range(len(self.balances)):
            torch.cuda.set_device(self.devices[i])
            hidden_states = hidden_states.to(self.devices[i], non_blocking=True)
            pre_padding_len = pre_padding_len.to(self.devices[i], non_blocking=True)
            for _ in range(self.balances[i]):
                layer = self.partitions[layer_id]
                # embedding
                if layer_id == 0:
                    hidden_states = layer(hidden_states, position_ids, token_type_ids)
                else:
                    hidden_states = layer(
                        hidden_states,
                        encoder_outputs=encoder_outputs,
                        first_pass=first_pass,
                        pre_padding_len=pre_padding_len,
                        head_mask=None,
                        reorder_state=self.reorder_state,
                        normalize_before=True,
                    )
                layer_id += 1
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


    @staticmethod
    def from_pretrained(model_id_or_path: str,max_batch, full_seq_len, data_type, balances=None):
        """EET pipeline parallel gpt2 model."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        layernorm_dict = {}
        torch_model = GPT2Model.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        model_name = "gpt2"

        for k, v in torch_model.state_dict().items():
            if 'e.' in k:
                embedding_dict[k] = v
            if 'h.' in k:
                k = convert_name(k, "gpt2")
                k = k[k.find('layer.'):]
                if 'ffn.intermediate.weight' in k or 'ffn.output.weight' in k:
                    model_dict[k] = torch.t(v).contiguous()
                else:
                    model_dict[k] = v
            if 'ln_f' in k:
                layernorm_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        configs = []
        devices = []
        partitions = []
        activation_fn = cfg.activation_function
        batch_size = max_batch
        full_seq_len = full_seq_len
        layer_num = cfg.n_layer + 1

        # init eet config
        if isinstance(balances, List):
            assert sum(balances) == layer_num, "sum of balances is not equal to module layer num"
        else:
            print("input balance is not a list, so run with default device")
            balances = [cfg.n_layer + 1]
        for i in range(len(balances)):
            device = f"cuda:{i}"
            config = meta_desc(batch_size, cfg.n_head, cfg.n_embd, cfg.n_layer,
                               cfg.n_positions, full_seq_len, data_type, device, False,
                               activation_fn)
            configs.append(config)
            devices.append(device)

        # TODO partitions
        layer_id = 0
        for i in range(len(balances)):
            tmp = []
            for _ in range(balances[i]):
                # embedding
                if layer_id == 0:
                    partitions.append(EETGPT2Embedding.from_torch(configs[0], embedding_dict, data_type, device=devices[0]))
                # encoder layer
                else:
                    partitions.append(EETDecoderLayer.from_torch(configs[i], layer_model_dict['layer.' + str(layer_id-1)], layer_id-1, data_type=data_type, add_cross_attn=False, bias=True, is_standard=False, device=devices[i]))
                layer_id += 1
        layer_norm = EETLayerNorm.from_torch(configs[-1], layernorm_dict['ln_f.weight'], layernorm_dict['ln_f.bias'], data_type=data_type, device=devices[-1])
        eet_model = EETPipeGPTModel(cfg, partitions, layer_norm, balances, devices)
        return eet_model

    def from_torch(torch_model, max_batch, full_seq_len, data_type, balances=None):
        """EET pipeline parallel gpt2 model."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        layernorm_dict = {}
        cfg = torch_model.config
        model_name = "gpt2"

        for k, v in torch_model.state_dict().items():
            if 'e.' in k:
                embedding_dict[k] = v
            if 'h.' in k:
                k = convert_name(k, "gpt2")
                k = k[k.find('layer.'):]
                if 'ffn.intermediate.weight' in k or 'ffn.output.weight' in k:
                    model_dict[k] = torch.t(v).contiguous()
                else:
                    model_dict[k] = v
            if 'ln_f' in k:
                layernorm_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        configs = []
        devices = []
        partitions = []
        activation_fn = cfg.activation_function
        batch_size = max_batch
        full_seq_len = full_seq_len
        layer_num = cfg.n_layer + 1

        # init eet config
        if isinstance(balances, List):
            assert sum(balances) == layer_num, "sum of balances is not equal to module layer num"
        else:
            print("input balance is not a list, so run with default device")
            balances = [cfg.n_layer + 1]
        for i in range(len(balances)):
            device = f"cuda:{i}"
            config = meta_desc(batch_size, cfg.n_head, cfg.n_embd, cfg.n_layer,
                               cfg.n_positions, full_seq_len, data_type, device, False,
                               activation_fn)
            configs.append(config)
            devices.append(device)

        # TODO partitions
        layer_id = 0
        for i in range(len(balances)):
            tmp = []
            for _ in range(balances[i]):
                # embedding
                if layer_id == 0:
                    partitions.append(EETGPT2Embedding.from_torch(configs[0], embedding_dict, data_type, device=devices[0]))
                # encoder layer
                else:
                    partitions.append(EETDecoderLayer.from_torch(configs[i], layer_model_dict['layer.' + str(layer_id-1)], layer_id-1, data_type=data_type, add_cross_attn=False, bias=True, is_standard=False, device=devices[i]))
                layer_id += 1
        layer_norm = EETLayerNorm.from_torch(configs[-1], layernorm_dict['ln_f.weight'], layernorm_dict['ln_f.bias'], data_type=data_type, device=devices[-1])
        eet_model = EETPipeGPTModel(cfg, partitions, layer_norm, balances, devices)
        return eet_model

