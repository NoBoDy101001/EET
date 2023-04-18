#ifndef _OP_T5FFN_HPP_
#define _OP_T5FFN_HPP_

#include "op/common.hpp"
#include "op/meta_desc.hpp"
#include "op/mmanager.hpp"

namespace eet{
    namespace op{

        class T5FeedForwardNetwork : public OpBase{
        public:
            T5FeedForwardNetwork(MetaDesc& desc,
                            const torch::Tensor& Intermediate_0_weights,
                            const torch::Tensor& Intermediate_0_bias,
                            const torch::Tensor& Intermediate_1_weights,
                            const torch::Tensor& Intermediate_1_bias,
                            const torch::Tensor& Output_weights,
                            const torch::Tensor& Output_bias,
                            const torch::Tensor& layernorm_weights,
                            const torch::Tensor& layernorm_bias,
                            const std::string &ffn_cache_name = "out_cache");

            torch::Tensor forward(torch::Tensor& input, 
                                    bool pre_layernorm,
                                    bool add_residual);

            ~T5FeedForwardNetwork(){
            };
            
        private:
            void layer_norm(const torch::Tensor& input_tensor, Buffer& layernorm_tensor);

            void fc1_mul(void* input,  Buffer& ffn_inner);

            void fc2_mul(void* input,  Buffer& ffn_inner);

            void gated_gelu(Buffer& inner_gelu, Buffer& inner_linear);

            void add_bias_act(Buffer& ffn_inner);

            void fc3_mul(const Buffer& ffn_inner, Buffer& output, torch::Tensor& input_tensor);

            void add_input_bias_layernorm(Buffer& output,torch::Tensor& input_tensor,bool pre_layernorm, bool add_residual);

            MetaDesc& desc_;
            // torch::Tensor output_;

            std::vector<int> fc1_mnk_ = {0, 0, 0}, fc2_mnk_ = {0, 0, 0}, fc3_mnk_ = {0, 0, 0};
            std::pair<bool, cublasLtMatmulAlgo_t> fc1_algo_, fc2_algo_, fc3_algo_;
            int act_type_;
            int cur_batch_size_;
            int cur_seq_len_;
            int size_per_head_;
            int d_ff_;
            int workspace_size_ = 0;

            void* alpha_;
            void* beta_;
            void* fc3_beta_;

            void* intermediate_0_weights_;
            void* intermediate_0_bias_;
            void* intermediate_1_weights_;
            void* intermediate_1_bias_;
            void* output_weights_;
            void* output_bias_;
            void* layernorm_weights_;
            void* layernorm_bias_;
            void* workspace_ = nullptr;
            std::string ffn_cache_name_;
        };
    }
}
#endif