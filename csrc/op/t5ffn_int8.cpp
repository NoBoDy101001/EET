#include "op/t5ffn_int8.hpp"
#include "op/cublasLt_wrappers.hpp"
#include "core/add_bias.cuh"
#include "core/layer_norm.cuh"
#include "core/gpt2_self_softmax.cuh"
#include "core/activation_kernel.cuh"

#define use_autotune true
namespace eet
{
    namespace op
    {
        T5FeedForwardNetworkInt8::T5FeedForwardNetworkInt8(MetaDesc& desc,
                            const torch::Tensor& Intermediate_0_weights,
                            const torch::Tensor& Intermediate_0_bias,
                            const torch::Tensor& Intermediate_1_weights,
                            const torch::Tensor& Intermediate_1_bias,
                            const torch::Tensor& Output_weights,
                            const torch::Tensor& Output_bias,
                            const torch::Tensor& layernorm_weights,
                            const torch::Tensor& layernorm_bias,
                            const std::string& ffn_cache_name) : 
        desc_(desc),
        intermediate_0_weights_(Intermediate_0_weights.data_ptr()),
        intermediate_0_bias_(Intermediate_0_bias.data_ptr()),
        intermediate_1_weights_(Intermediate_1_weights.data_ptr()),
        intermediate_1_bias_(Intermediate_1_bias.data_ptr()),
        output_weights_(Output_weights.data_ptr()),
        output_bias_(Output_bias.data_ptr()),
        layernorm_weights_(layernorm_weights.data_ptr()),
        layernorm_bias_(layernorm_bias.data_ptr()),
        ffn_cache_name_(ffn_cache_name)
        {   
            // Currently only supports gelu and relu
            if (desc_.activation_fn_ == "quick_gelu")
            {
                act_type_ = 2;
            }
            else if (desc_.activation_fn_ == "gelu" || desc_.activation_fn_ == "gelu_new" || desc_.activation_fn_ == "gelu_fast")
            {
                act_type_ = 1;
            }
            else if(desc_.activation_fn_ == "relu")
            {
                // relu
                act_type_ = 0;
            }
            else
            {
                std::cout << "unsupported activation " << std::endl;
                return;
            }
            if (desc_.d_ff_ == 0) {
                size_per_head_ = desc_.hidden_units_ / desc_.head_num_;
                d_ff_ = desc_.hidden_units_ * 4;
            } else {
                size_per_head_ = desc_.d_kv_;
                d_ff_ = desc_.d_ff_;
            }
            MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, ffn_cache_name_);
            MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.max_seq_len_ * d_ff_, desc_.dtype_, desc_.options_, "t5ffn_buffer1");
            MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.max_seq_len_ * d_ff_, desc_.dtype_, desc_.options_, "t5ffn_buffer2");
            check_cuda_error(cudaMalloc((void **)&int8_intermediate_0_weights_, sizeof(int8_t) * desc_.hidden_units_ * d_ff_));
            check_cuda_error(cudaMalloc((void **)&int8_intermediate_1_weights_, sizeof(int8_t) * desc_.hidden_units_ * d_ff_));
            check_cuda_error(cudaMalloc((void **)&int8_output_weights_, sizeof(int8_t) * d_ff_ * desc_.hidden_units_));
            transform_int8_weight(desc_.ltHandle, (int8_t* )intermediate_0_weights_, int8_intermediate_0_weights_, d_ff_, desc_.hidden_units_);
            transform_int8_weight(desc_.ltHandle, (int8_t* )intermediate_1_weights_, int8_intermediate_1_weights_, d_ff_, desc_.hidden_units_);
            transform_int8_weight(desc_.ltHandle, (int8_t* )output_weights_, int8_output_weights_, desc_.hidden_units_, d_ff_);

            switch (desc_.dtype_)
            {
            case torch::kFloat32:
                alpha_ = new float();
                beta_ = new float();
                *((float *)alpha_) = 1.0f;
                *((float *)beta_) = 0.0f;
                break;
            case torch::kFloat16:
                alpha_ = new half();
                beta_ = new half();
                *((half *)alpha_) = (half)1.0f;
                *((half *)beta_) = (half)0.0f;
                break;
            case torch::kBFloat16:
                alpha_ = new float();
                beta_ = new float();
                *((float *)alpha_) = 1.0f;
                *((float *)beta_) = 0.0f;
                break;
            //TODO
            case torch::kInt8:
                break;
            }
        }

        torch::Tensor T5FeedForwardNetworkInt8::forward(torch::Tensor &input,
                                                    bool pre_layernorm,
                                                    bool add_residual)
        {
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as T5FeedForwardNetworkInt8's dtype");
            cur_batch_size_ = input.sizes()[0];
            cur_seq_len_ = input.sizes()[1];

            //ffn_inner
            Buffer &ffn_inner_gelu = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ * d_ff_, desc_.dtype_, desc_.options_);
            Buffer &ffn_inner_linear = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ * d_ff_, desc_.dtype_, desc_.options_);

            // pre_layerNorm
            Buffer& layernorm_tensor = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_);
            layer_norm(input, layernorm_tensor);

            fc1_mul(layernorm_tensor.data_ptr(), ffn_inner_gelu);
            fc2_mul(layernorm_tensor.data_ptr(), ffn_inner_linear);

            layernorm_tensor.free();
            gated_gelu(ffn_inner_gelu, ffn_inner_linear);
            ffn_inner_linear.free();

            Buffer &output = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, ffn_cache_name_);

            fc3_mul(ffn_inner_gelu, output);

            ffn_inner_gelu.free();

            add_input_bias_layernorm(output, input, pre_layernorm, add_residual);

            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);
            return std::move(res);
        }

        // layerNorm
        void T5FeedForwardNetworkInt8::layer_norm(const torch::Tensor& input_tensor, Buffer& layernorm_tensor)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_;
            if (layernorm_bias_ != nullptr) {
                RUN_KERNEL(layernorm,desc_.dtype_,input_tensor.data_ptr(),layernorm_weights_,layernorm_bias_,layernorm_tensor.data_ptr(), m, n, desc_.stream);
            } else {
                RUN_KERNEL(T5layernorm,desc_.dtype_,input_tensor.data_ptr(),layernorm_weights_,layernorm_tensor.data_ptr(), m, n, desc_.stream);
            }
            
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void T5FeedForwardNetworkInt8::fc1_mul(void* input, Buffer &ffn_inner)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = desc_.hidden_units_ ;
            int n = d_ff_;

            cublasLtIgemm(desc_.ltHandle,
                          m, n, k,
                          alpha_,
                          (int8_t* )input,
                          int8_intermediate_0_weights_,
                          beta_,
                          (int32_t* )ffn_inner.data_ptr(),
                          desc_.dataType_,
                          desc_.scaleType_,
                          desc_.computeType_,
                          workspace_, 0, desc_.stream);

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void T5FeedForwardNetworkInt8::add_bias_act(Buffer& ffn_inner)
        {
            int m = cur_batch_size_ * cur_seq_len_;
            int n = d_ff_;
            
            RUN_KERNEL(add_bias_act_kernel,desc_.dtype_,ffn_inner.data_ptr(), intermediate_0_bias_, m, n, act_type_ ,desc_.stream)
            
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void T5FeedForwardNetworkInt8::gated_gelu(Buffer& inner_gelu, Buffer& inner_linear)
        {
            int m = cur_batch_size_ * cur_seq_len_;
            int n = d_ff_;
            
            RUN_KERNEL(gated_gelu_kernel, desc_.dtype_, inner_gelu.data_ptr(), inner_linear.data_ptr(), m, n, act_type_ ,desc_.stream)
            
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void T5FeedForwardNetworkInt8::fc2_mul(void* input, Buffer &ffn_inner)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = desc_.hidden_units_ ;
            int n = d_ff_;

            cublasLtIgemm(desc_.ltHandle,
                          m, n, k,
                          alpha_,
                          (int8_t* )input,
                          int8_intermediate_1_weights_,
                          beta_,
                          (int32_t* )ffn_inner.data_ptr(),
                          desc_.dataType_,
                          desc_.scaleType_,
                          desc_.computeType_,
                          workspace_, 0, desc_.stream);

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void T5FeedForwardNetworkInt8::fc3_mul(const Buffer& ffn_inner, Buffer& output)
        { 
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_ ;
            int k = d_ff_;

            cublasLtIgemm(desc_.ltHandle,
                          m, n, k,
                          alpha_,
                          (int8_t* )ffn_inner.data_ptr(),
                          int8_output_weights_,
                          beta_,
                          (int32_t* )output.data_ptr(),
                          desc_.dataType_,
                          desc_.scaleType_,
                          desc_.computeType_,
                          workspace_, 0, desc_.stream);
            

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void T5FeedForwardNetworkInt8::add_input_bias_layernorm(Buffer& output,torch::Tensor& input,bool pre_layernorm, bool add_residual)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_ ;
            int k = d_ff_;

            if(add_residual)
            {   
                if(!pre_layernorm)
                {   
                    // add_bias + add_residual + layer_norm
                    RUN_KERNEL(add_bias_input_layernorm_kernel,desc_.dtype_,
                                        output.data_ptr(),input.data_ptr(), 
                                        output_bias_,layernorm_weights_,
                                        layernorm_bias_,m , n, desc_.stream);
                }
                else
                {
                    // add_bias + add_residual
                    RUN_KERNEL(add_bias_input_kernel, desc_.dtype_, output.data_ptr(), input.data_ptr(),output_bias_, m , n, desc_.stream);
                }
            }
            else
            {
                // only add bias
                if (output_bias_ != nullptr) {
                    RUN_KERNEL(add_bias_kernel, desc_.dtype_, output.data_ptr(), output_bias_,m , n, desc_.stream);
                }
            }
        }

        void T5FeedForwardNetworkInt8::transform_int8_weight(cublasLtHandle_t ltHandle, const int8_t* input_weight, int8_t* output_weight, int row, int col)
        {
            cublasLtOrder_t order_ROW = CUBLASLT_ORDER_ROW;
            cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
            cublasLtMatrixTransformDesc_t transformDesc = NULL;
            cublasLtMatrixLayout_t input_desc = NULL, output_desc = NULL;

            int ldbtransform = 32 * roundoff(row, 8);
            float transformAlpha = 1.0f, transformBeta = 0.0f;
            check_cuda_error(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

            // transform fc0 layout
            cublasLtMatrixLayoutCreate(&input_desc, CUDA_R_8I, row, col, col);
            cublasLtMatrixLayoutSetAttribute(input_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_ROW, sizeof(order_ROW));
            cublasLtMatrixLayoutCreate(&output_desc, CUDA_R_8I, row, col, ldbtransform);
            cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, input_weight, input_desc, &transformBeta, NULL, NULL, output_weight, output_desc, 0);
            cublasLtMatrixLayoutDestroy(input_desc);
            cublasLtMatrixLayoutDestroy(output_desc);
            cublasLtMatrixTransformDescDestroy(transformDesc);
        }
    } // namespace op
} // namespace eet
