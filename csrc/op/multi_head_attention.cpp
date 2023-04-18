#include "op/multi_head_attention.hpp"
#include "op/cublasLt_wrappers.hpp"
#include "core/add_bias.cuh"
#include "core/transpose.cuh"
#include "core/layer_norm.cuh"
#include "core/self_add_bias.cuh"
#include "core/attention_dispatch.cuh"
#include "core/bert_softmax.cuh"
#include "core/pre_process.cuh"
#include <iostream>

// for gpt
namespace eet{
    namespace op{
        MultiHeadAttention::MultiHeadAttention(MetaDesc& desc,
                                    const torch::Tensor& QKV_weights,
                                    const torch::Tensor& Q_bias,
                                    const torch::Tensor& K_bias,
                                    const torch::Tensor& V_bias,
                                    const torch::Tensor& Output_weights,
                                    const torch::Tensor& Output_bias,
                                    const torch::Tensor& layernorm_weights,
                                    const torch::Tensor& layernorm_bias):
            desc_(desc),
            max_len_(0),
            qkv_weights_(QKV_weights.data_ptr()),
            q_bias_(Q_bias.data_ptr()),
            k_bias_(K_bias.data_ptr()),
            v_bias_(V_bias.data_ptr()),
            output_weights_(Output_weights.data_ptr()),
            output_bias_(Output_bias.data_ptr()),
            layernorm_weights_(layernorm_weights.data_ptr()),
            layernorm_bias_(layernorm_bias.data_ptr())
        {   
            with_bias_ = q_bias_ != nullptr ? true : false;
            if (desc_.d_kv_ == 0) {
                size_per_head_ = desc_.hidden_units_ / desc_.head_num_;
                inner_dim_ = desc_.hidden_units_;
            } else {
                size_per_head_ = desc_.d_kv_;
                inner_dim_ = size_per_head_ * desc_.head_num_;
            }
            MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, "self_attn_cache");
            MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.max_seq_len_ * inner_dim_ * 3, desc_.dtype_, desc_.options_, "qkv_full");
            MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, "layernorm");
            MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.max_seq_len_ * inner_dim_, desc_.dtype_, desc_.options_, "q_buf");
            MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.max_seq_len_ * inner_dim_, desc_.dtype_, desc_.options_, "k_buf");
            MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.max_seq_len_ * inner_dim_, desc_.dtype_, desc_.options_, "v_buf");
            MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.head_num_ * desc_.max_seq_len_ * desc_.max_seq_len_, desc_.dtype_, desc_.options_, "qk_buf");
// #ifdef _AUTOTUNE_
#ifdef VERSION_INFO
            torch::TensorOptions tmp = torch::TensorOptions().dtype(torch::kInt8).device("cuda:0").requires_grad(false);
            Buffer &workspace = MManager::get_instance().get_cache(WORKSPACE_SIZE, torch::kInt8, tmp, "workspace");
            workspace_ = workspace.data_ptr();
            workspace_size_ = WORKSPACE_SIZE;
#endif
            switch (desc_.dtype_)
            {
            case torch::kFloat32:
                alpha_ = new float();
                beta_ = new float();
                atten_scaler_ = new float();
                o_beta_ = new float();
                *((float *)alpha_) = 1.0f;
                *((float *)beta_) = 0.0f;
                *((float *)o_beta_) = 1.0f;
                if (with_bias_) {
                    *((float *)atten_scaler_) = sqrt(1.0f / size_per_head_);
                } else {
                    *((float *)atten_scaler_) = 1.0f;
                }
                break;
            case torch::kFloat16:
                alpha_ = new half();
                beta_ = new half();
                atten_scaler_ = new half();
                o_beta_ = new half();
                *((half *)alpha_) = (half)1.0f;
                *((half *)beta_) = (half)0.0f;
                *((half *)o_beta_) = (half)1.0f;
                if (with_bias_) {
                    *((half *)atten_scaler_) = sqrt(1.0f / size_per_head_);
                } else {
                    *((half *)atten_scaler_) = 1.0f;
                }    
                break;
            case torch::kBFloat16:
                alpha_ = new float();
                beta_ = new float();
                atten_scaler_ = new float();
                o_beta_ = new float();
                *((float *)alpha_) = 1.0f;
                *((float *)beta_) = 0.0f;
                *((float *)o_beta_) = 1.0f;
                if (with_bias_) {
                    *((float *)atten_scaler_) = sqrt(1.0f / size_per_head_);
                } else {
                    *((float *)atten_scaler_) = 1.0f;
                }
                break;
            //TODO
            case torch::kInt8:
                break;
            }
        }

        // encoder
        torch::Tensor MultiHeadAttention::forward(
                                    torch::Tensor& input,
                                    const torch::Tensor& pre_padding_len,
                                    bool pre_layernorm,
                                    bool add_residual,
                                    bool need_sequence_mask,
                                    const torch::Tensor &relative_attention_bias){
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as MultiHeadAttention's dtype");
            cur_batch_size_ = input.sizes()[0];
            cur_seq_len_ = input.sizes()[1];
            assert((cur_batch_size_ <= desc_.batch_size_)&& "cur_batch_size_ must be less than or equal to max_batch_size_");

            Buffer& qkv_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_ * 3, desc_.dtype_, desc_.options_);
            if(pre_layernorm)
            {
                // pre_layerNorm
                Buffer& layernormed_query = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                        desc_.hidden_units_, desc_.dtype_, desc_.options_);
                layer_norm(input,layernormed_query);

                //qkv * weights
                qkv_weights_mul(layernormed_query.data_ptr(), qkv_buffer);
                layernormed_query.free();
            }
            else{
                //qkv * weights
                qkv_weights_mul(input.data_ptr(), qkv_buffer);
            }

            //qkv add bias                
            Buffer& q_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);
            Buffer& k_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);
            Buffer& v_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);
            qkv_add_bias(qkv_buffer, q_buf, k_buf, v_buf);
            
            qkv_buffer.free();


            //q * k
            Buffer& qk_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.head_num_ *
                                        desc_.max_seq_len_ * desc_.max_seq_len_, desc_.dtype_, desc_.options_);
            q_k_mul(q_buf, k_buf, qk_buf);

            q_buf.free();

            // relative attention bias
            void* relative_attention_bias_ = relative_attention_bias.data_ptr();

            //softmax
            const int64_t *padding_len = pre_padding_len.data_ptr<int64_t>();

            qk_softmax(qk_buf, relative_attention_bias_, padding_len, need_sequence_mask);

            //attn * v
            Buffer& transpose_dst = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);
            
            attn_v_mul(qk_buf,v_buf,transpose_dst);

            qk_buf.free();
            k_buf.free();
            v_buf.free();

            //transpose
            Buffer& dst = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);

            transpose(transpose_dst, dst);
            transpose_dst.free();

            //project
            Buffer &output = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, "self_attn_cache");

            project(dst,output, input, pre_layernorm, add_residual);
            dst.free();
            // output_ = output_[input.sizes()];

            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);

            return std::move(res);
        }

        // layerNorm
        void MultiHeadAttention::layer_norm(const torch::Tensor& input, Buffer& layernorm_query)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_;
            if (with_bias_) {
                RUN_KERNEL(layernorm,desc_.dtype_,input.data_ptr(),layernorm_weights_,layernorm_bias_,layernorm_query.data_ptr(), m, n, desc_.stream);
            } else {
                RUN_KERNEL(T5layernorm,desc_.dtype_,input.data_ptr(),layernorm_weights_,layernorm_query.data_ptr(), m, n, desc_.stream);
            }
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::qkv_weights_mul(void* input, 
                                    Buffer& qkv_buffer){
            const int m = cur_batch_size_ * cur_seq_len_;
            const int k = desc_.hidden_units_;
            const int n = 3 * inner_dim_;

            if (qkv_mnk_ != std::vector<int> {n, m, k}) {
                qkv_algo_ = desc_.getAlgo(n, m, k);
                qkv_mnk_ = std::vector<int> {n, m, k};
            }
#ifdef _AUTOTUNE_
            qkv_algo_.first = true;
#endif
            if (qkv_algo_.first) {
                cublasLtgemm(desc_.ltHandle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            n, m, k,
                            alpha_,
                            qkv_weights_, n,
                            input, k,
                            beta_,
                            qkv_buffer.data_ptr(), n,
                            desc_.dataType_,
                            desc_.scaleType_,
                            desc_.computeType_, qkv_algo_.second,
                            workspace_, workspace_size_, desc_.stream);
            } else {
                check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    alpha_,
                    qkv_weights_, desc_.dataType_, n,
                    input, desc_.dataType_, k,
                    beta_,
                    qkv_buffer.data_ptr(), desc_.dataType_, n,
                    desc_.computeType_,
                    CUBLAS_GEMM_DEFAULT));
            }
#ifdef _AUTOTUNE_
            desc_.saveAlgo(n, m, k, qkv_algo_.second);
#endif

#ifdef _DEBUG_MODE_
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::qkv_add_bias(const Buffer &qkv_buffer,
                                                       Buffer &q_buf,
                                                       Buffer &k_buf,
                                                       Buffer &v_buf)
        {
            RUN_KERNEL(fused_add_QKV_bias_kernel,desc_.dtype_,qkv_buffer.data_ptr(),q_bias_,
                       k_bias_, v_bias_, q_buf.data_ptr(), k_buf.data_ptr(), v_buf.data_ptr(),
                       cur_batch_size_, cur_seq_len_, desc_.head_num_, size_per_head_, desc_.stream);

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::q_k_mul(const Buffer& q_buf, const Buffer& k_buf,
                                                Buffer& qk_buf){
            check_cuda_error(cublasGemmStridedBatchedEx(desc_.cublasHandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                cur_seq_len_, cur_seq_len_, size_per_head_,
                atten_scaler_,
                k_buf.data_ptr(), desc_.dataType_, size_per_head_, cur_seq_len_ * size_per_head_,
                q_buf.data_ptr(), desc_.dataType_, size_per_head_, cur_seq_len_ * size_per_head_,
                beta_,
                qk_buf.data_ptr(), desc_.dataType_, cur_seq_len_, cur_seq_len_ * cur_seq_len_,
                cur_batch_size_ * desc_.head_num_,
                desc_.computeType_,
                CUBLAS_GEMM_DEFAULT));

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::add_relative_attn_bias(Buffer &qk_buf, void* relative_attention_bias)
        {
            RUN_KERNEL(add_relative_attn_bias_kernel, desc_.dtype_, qk_buf.data_ptr(), relative_attention_bias,
                       cur_batch_size_, desc_.head_num_, cur_seq_len_, desc_.stream);

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::qk_softmax(Buffer &qk_buf, void* relative_attention_bias, const int64_t *padding_len, bool need_sequence_mask) {
            // float scalar = 1 / sqrtf(size_per_head_ * 1.0f);
            RUN_KERNEL(bert_softmax_kernel, desc_.dtype_, qk_buf.data_ptr(), relative_attention_bias, padding_len, cur_batch_size_,
                       desc_.head_num_, cur_seq_len_, need_sequence_mask, desc_.stream);
#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::attn_v_mul(const Buffer& qk_buf,
                                             const Buffer& v_buf,
                                             Buffer& transpose_dst){
            check_cuda_error(cublasGemmStridedBatchedEx(desc_.cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    size_per_head_, cur_seq_len_, cur_seq_len_,
                    alpha_,
                    v_buf.data_ptr(), desc_.dataType_, size_per_head_, cur_seq_len_ * size_per_head_,
                    qk_buf.data_ptr(), desc_.dataType_, cur_seq_len_, cur_seq_len_ * cur_seq_len_,
                    beta_,
                    transpose_dst.data_ptr(), desc_.dataType_, size_per_head_, cur_seq_len_ * size_per_head_,
                    cur_batch_size_ * desc_.head_num_,
                    desc_.computeType_,
                    CUBLAS_GEMM_DEFAULT));

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void MultiHeadAttention::transpose(const Buffer& transpose_dst, Buffer&  dst){
            RUN_KERNEL(transpose_kernel,desc_.dtype_,transpose_dst.data_ptr(),dst.data_ptr(), cur_batch_size_, cur_seq_len_,
                    desc_.head_num_, size_per_head_, desc_.stream);

            #ifdef _DEBUG_MODE_
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
            #endif
        }

        void MultiHeadAttention::project(const Buffer& dst, Buffer& res, torch::Tensor& input, bool pre_layernorm, bool add_residual){
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = inner_dim_;
            int n = desc_.hidden_units_;

            if (o_mnk_ != std::vector<int> {n, m, k}) {
                o_algo_ = desc_.getAlgo(n, m, k);
                o_mnk_ = std::vector<int> {n, m, k};
            }
#ifdef _AUTOTUNE_
            o_algo_.first = true;
#endif
            if (o_algo_.first) {
                cublasLtgemm(desc_.ltHandle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            n, m, k,
                            alpha_,
                            output_weights_, n,
                            dst.data_ptr(), k,
                            o_beta_,
                            res.data_ptr(), n,
                            desc_.dataType_,
                            desc_.scaleType_,
                            desc_.computeType_, o_algo_.second,
                            workspace_, workspace_size_, 
                            desc_.stream, input.data_ptr());
            } else {
                check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N,
                                                n, m, k,
                                                alpha_,
                                                output_weights_, desc_.dataType_, n,
                                                dst.data_ptr(), desc_.dataType_, k,
                                                beta_,
                                                res.data_ptr(), desc_.dataType_, n,
                                                desc_.computeType_,
                                                CUBLAS_GEMM_DEFAULT));
            }

#ifdef _AUTOTUNE_
            desc_.saveAlgo(n, m, k, o_algo_.second);
#endif

            if(add_residual && !o_algo_.first)
            {   
                if(!pre_layernorm)
                {   
                    // add_bias + add_residual + layer_norm
                    RUN_KERNEL(add_bias_input_layernorm_kernel, desc_.dtype_,
                               res.data_ptr(), input.data_ptr(),
                               output_bias_, layernorm_weights_,
                               layernorm_bias_, m, n, desc_.stream);
                }
                else
                {
                    // add_bias + add_residual
                    RUN_KERNEL(add_bias_input_kernel, desc_.dtype_, res.data_ptr(), input.data_ptr(), output_bias_,
                               m, n, desc_.stream);
                }
            }
            else
            {
                // only add bias
                if (with_bias_)
                {
                    RUN_KERNEL(add_bias_kernel, desc_.dtype_, res.data_ptr(), output_bias_, m, n, desc_.stream);
                }
            }
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
         }
    }
}