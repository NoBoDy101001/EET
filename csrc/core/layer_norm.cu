#include "core/common.cuh"
#include <assert.h>

// layernorm code modified from Nvidia's DeepLearningExamples
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.1/fastertransformer/cuda/open_decoder.cu#L1369-L81429

template <typename T>
__global__ 
void add_bias_input_layernorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  local_out += (float)(out[blockIdx.x * n + tid] + input[blockIdx.x * n + tid] + __ldg(&bias[tid]));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  out[blockIdx.x * n + tid] = 
	    (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}

template <>
__global__ 
void add_bias_input_layernorm(half* out, const half* input, const half* bias, const half* gamma, const half* beta, int m, int n)
{

  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  const half2* bias_ptr = (const half2*)bias;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;

  float local_out = 0.0f;
  int id = blockIdx.x * n / 2 + tid; 
  local_out_fp2 = __half22float2(__hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg(&bias_ptr[tid])));
  local_out += local_out_fp2.x;
  local_out += local_out_fp2.y;

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
  variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  variance = blockReduceSum<float>(variance);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
  float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
  local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}

template <typename T>
__global__ 
void add_input_T5layernorm(T* out, const T* input, const T* gamma, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_variance;
  float variance = 0.0f;

  float local_out = 0.0f;
  local_out += (float)(out[blockIdx.x * n + tid] + input[blockIdx.x * n + tid]);

  variance = blockReduceSum<float>(local_out * local_out);
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  out[blockIdx.x * n + tid] = 
	    (T)((local_out * rsqrtf(s_variance)) * (float)(__ldg(&gamma[tid])));
}

template <>
__global__ 
void add_input_T5layernorm(half* out, const half* input, const half* gamma, int m, int n)
{
  int tid = threadIdx.x;
  __shared__ float s_variance;
  float variance = 0.0f;
  float2 local_out_fp2;

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  const half2* gamma_ptr = (const half2*)gamma;

  int id = blockIdx.x * n / 2 + tid; 
  local_out_fp2 = __half22float2(__hadd2(out_ptr[id], input_ptr[id]));

  variance = local_out_fp2.x * local_out_fp2.x;
  variance += local_out_fp2.y * local_out_fp2.y;
  variance = blockReduceSum<float>(variance);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
  local_out_fp2.x = (local_out_fp2.x) * s_variance * gamma_val.x;
  local_out_fp2.y = (local_out_fp2.y) * s_variance * gamma_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}


template <typename T>
__global__ void add_bias_input_layernorm_v2(T *out, const T *__restrict input, const T *__restrict bias,
                                            const T *__restrict gamma, const T *__restrict beta, int n)
{
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float local_out[ite];

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    local_out[i] = (float)(out[id] + __ldg(&input[id]) + __ldg(&bias[col_id]));
    sum += local_out[i];
  }

  mean = blockReduceSum<float>(sum);
  if (tid == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    float diff = local_out[i] - s_mean;
    var += diff * diff;
  }

  variance = blockReduceSum<float>(var);
  if (tid == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    out[id] = (T)((local_out[i] - s_mean) * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
  }
}

template <>
__global__ void add_bias_input_layernorm_v2(half *out, const half *__restrict input, const half *__restrict bias,
                                            const half *__restrict gamma, const half *__restrict beta, int n)
{
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  half2 local_out_half2[ite];

  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;
  const half2 *gamma_ptr = (const half2 *)gamma;
  const half2 *beta_ptr = (const half2 *)beta;

  // float sum = 0.0f;
  half2 sum = __float2half2_rn(0.0f);
#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    local_out_half2[i] = out_ptr[id] + __ldg(&input_ptr[id]) + __ldg(&bias_ptr[col_id]);
    sum += local_out_half2[i];
  }

  mean = blockReduceSum<float>((float)(sum.x + sum.y));
  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
  half2 s_mean_2 = __float2half2_rn(s_mean);
#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    local_out_half2[i] = local_out_half2[i] - s_mean_2;
    float v1 = (float)local_out_half2[i].x;
    float v2 = (float)local_out_half2[i].y;
    var += v1 * v1 + v2 * v2;
  }

  variance = blockReduceSum<float>(var);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]) + __ldg(&beta_ptr[col_id]);
  }
}

template <typename T>
__global__ void add_input_T5layernorm_v2(T *out, const T *__restrict input, const T *__restrict gamma, int n)
{
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  __shared__ float s_variance;
  float variance = 0.0f;
  float local_out[ite];

#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    local_out[i] = (float)(out[id] + __ldg(&input[id]));
  }

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    float diff = local_out[i];
    var += diff * diff;
  }

  variance = blockReduceSum<float>(var);
  if (tid == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    out[id] = (T)(local_out[i] * s_variance * (float)__ldg(&gamma[col_id]));
  }
}

template <>
__global__ void add_input_T5layernorm_v2(half *out, const half *__restrict input, const half *__restrict gamma, int n)
{
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_variance;
  float variance = 0.0f;
  half2 local_out_half2[ite];

  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *gamma_ptr = (const half2 *)gamma;

  // float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    local_out_half2[i] = out_ptr[id] + __ldg(&input_ptr[id]);
  }

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    float v1 = (float)local_out_half2[i].x;
    float v2 = (float)local_out_half2[i].y;
    var += v1 * v1 + v2 * v2;
  }

  variance = blockReduceSum<float>(var);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
  for (int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]);
  }
}

template <int item_per_thread>
__global__ void decoder_norm1_kernel_opt(const float *__restrict input,
                                         const float *__restrict gamma,
                                         const float *__restrict beta,
                                         float *output,
                                         int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  //float local_out = tid < n ? (float)(__ldg(&input[blockIdx.x * n + tid])) : 0.0f;
  float local_out[item_per_thread];
  for (int i = 0; i < item_per_thread; i++)
  {
    local_out[i] = (tid * item_per_thread + i) < n ? (float)(__ldg(&input[blockIdx.x * n + tid * item_per_thread + i])) : 0.0f;
  }

  mean = blockReduceSum_opt<float, item_per_thread>(local_out);

  if (threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float tmp[item_per_thread];
  for (int i = 0; i < item_per_thread; i++)
  {
    tmp[i] = (tid * item_per_thread + i) < n ? (local_out[i] - s_mean) * (local_out[i] - s_mean) : 0.0f;
  }

  //要保证第二次归约能把所有的和算出来,这个item_per_thread需要设置的足够大
  variance = blockReduceSum_opt<float, item_per_thread>(tmp);

  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);

  __syncthreads();

  for (int i = 0; i < item_per_thread; i++)
  {
    if (tid * item_per_thread + i < n)
    {
      output[blockIdx.x * n + tid * item_per_thread + i] =
          (float)(((local_out[i] - s_mean) * s_variance) * (float)(__ldg(&gamma[tid * item_per_thread + i])) + (float)(__ldg(&beta[tid * item_per_thread + i])));
    }
  }
}

template <int item_per_thread>
__global__ void decoder_norm1_kernel_opt(const half *__restrict input,
                                         const half *__restrict gamma,
                                         const half *__restrict beta,
                                         half *output,
                                         int m, int n)
{
  const int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float2 local_out_fp2[item_per_thread];

  const half2 *input_ptr = (const half2 *)input;
  const half2 *gamma_ptr = (const half2 *)gamma;
  const half2 *beta_ptr = (const half2 *)beta;
  half2 *output_ptr = (half2 *)output;

  float local_out[item_per_thread];

  for (int i = 0; i < item_per_thread; i++)
  {
    local_out[i] = 0.0f;
  }

  for (int i = 0; i < item_per_thread; i++)
  {
    local_out_fp2[i] = (tid * item_per_thread + i) < n ? __half22float2((__ldg(&input_ptr[blockIdx.x * (n >> 1) + tid * item_per_thread + i]))) : make_float2(0.0f, 0.0f);
    local_out[i] += local_out_fp2[i].x;
    local_out[i] += local_out_fp2[i].y;
  }

  mean = blockReduceSum_opt<float, item_per_thread>(local_out);
  if (tid == 0)
    s_mean = mean / n;
  __syncthreads();

  float tmp[item_per_thread];
  for (int i = 0; i < item_per_thread; i++)
  {
    tmp[i] = (tid * item_per_thread + i) < n ? (local_out_fp2[i].x - s_mean) * (local_out_fp2[i].x - s_mean) +
                                                   (local_out_fp2[i].y - s_mean) * (local_out_fp2[i].y - s_mean)
                                             : 0.0f;
  }

  variance = blockReduceSum_opt<float, item_per_thread>(tmp);
  if (tid == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  for (int i = 0; i < item_per_thread; i++)
  {
    if (tid * item_per_thread + i < n)
    {
      float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid * item_per_thread + i]));
      float2 beta_val = __half22float2(__ldg(&beta_ptr[tid * item_per_thread + i]));
      local_out_fp2[i].x = (local_out_fp2[i].x - s_mean) * s_variance * gamma_val.x + beta_val.x;
      local_out_fp2[i].y = (local_out_fp2[i].y - s_mean) * s_variance * gamma_val.y + beta_val.y;
      output_ptr[blockIdx.x * (n >> 1) + tid * item_per_thread + i] = __float22half2_rn(local_out_fp2[i]);
    }
  }
}

template <int item_per_thread>
__global__ void decoder_norm1_kernel_opt(const nv_bfloat16 *__restrict input,
                                         const nv_bfloat16 *__restrict gamma,
                                         const nv_bfloat16 *__restrict beta,
                                         nv_bfloat16 *output,
                                         int m, int n)
{
  const int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float2 local_out_fp2[item_per_thread];

  const nv_bfloat162 *input_ptr = (const nv_bfloat162 *)input;
  const nv_bfloat162 *gamma_ptr = (const nv_bfloat162 *)gamma;
  const nv_bfloat162 *beta_ptr = (const nv_bfloat162 *)beta;
  nv_bfloat162 *output_ptr = (nv_bfloat162 *)output;

  float local_out[item_per_thread];

  for (int i = 0; i < item_per_thread; i++)
  {
    local_out[i] = 0.0f;
  }

  for (int i = 0; i < item_per_thread; i++)
  {
    local_out_fp2[i] = (tid * item_per_thread + i) < n ? __bfloat1622float2((__ldg(&input_ptr[blockIdx.x * (n >> 1) + tid * item_per_thread + i]))) : make_float2(0.0f, 0.0f);
    local_out[i] += local_out_fp2[i].x;
    local_out[i] += local_out_fp2[i].y;
  }

  mean = blockReduceSum_opt<float, item_per_thread>(local_out);
  if (tid == 0)
    s_mean = mean / n;
  __syncthreads();

  float tmp[item_per_thread];
  for (int i = 0; i < item_per_thread; i++)
  {
    tmp[i] = (tid * item_per_thread + i) < n ? (local_out_fp2[i].x - s_mean) * (local_out_fp2[i].x - s_mean) +
                                                   (local_out_fp2[i].y - s_mean) * (local_out_fp2[i].y - s_mean)
                                             : 0.0f;
  }

  variance = blockReduceSum_opt<float, item_per_thread>(tmp);
  if (tid == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  for (int i = 0; i < item_per_thread; i++)
  {
    if (tid * item_per_thread + i < n)
    {
      float2 gamma_val = __bfloat1622float2(__ldg(&gamma_ptr[tid * item_per_thread + i]));
      float2 beta_val = __bfloat1622float2(__ldg(&beta_ptr[tid * item_per_thread + i]));
      local_out_fp2[i].x = (local_out_fp2[i].x - s_mean) * s_variance * gamma_val.x + beta_val.x;
      local_out_fp2[i].y = (local_out_fp2[i].y - s_mean) * s_variance * gamma_val.y + beta_val.y;
      output_ptr[blockIdx.x * (n >> 1) + tid * item_per_thread + i] = __float22bfloat162_rn(local_out_fp2[i]);
    }
  }
}

template <typename T>
__global__ void T5norm_kernel_opt(const T *__restrict input,
                                  const T *__restrict gamma,
                                  T *output,
                                  int m, int n)
{
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float variance = 0.0f;
    float local_var_sum = 0.0f;
    const float4 *input_ptr = reinterpret_cast<const float4 *>(input) + blockIdx.x * n;
    float4 *output_ptr = (float4 *)(output) + blockIdx.x * n;

    for (int i = tid; i < n; i += blockDim.x) {
      float4 tmp = input_ptr[i];
      local_var_sum += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)(n << 2) + 1e-6);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
      float4 tmp = input_ptr[i];
      float4 gamma_tmp = __ldg(reinterpret_cast<const float4 *>(gamma) + i);
      tmp.x = tmp.x * s_variance * gamma_tmp.x;
      tmp.y = tmp.y * s_variance * gamma_tmp.y;
      tmp.z = tmp.z * s_variance * gamma_tmp.z;
      tmp.w = tmp.w * s_variance * gamma_tmp.w;
      output_ptr[i] = tmp;
    }
}

template <>
__global__ void T5norm_kernel_opt(const __half *__restrict input,
                                  const __half *__restrict gamma,
                                  __half *output,
                                  int m, int n)
{
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float variance = 0.0f;
    float local_var_sum = 0.0f;
    const float4 *input_ptr = reinterpret_cast<const float4 *>(input) + blockIdx.x * n;
    float4 *output_ptr = (float4 *)(output) + blockIdx.x * n;

    for (int i = tid; i < n; i += blockDim.x) {
      float4 tmp = input_ptr[i];
      __half2 *val_h2 = (__half2 *)(&tmp);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 val_f2 = __half22float2(val_h2[j]);
        local_var_sum += val_f2.x * val_f2.x + val_f2.y * val_f2.y;
      }
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)(n << 3) + 1e-6);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
      float4 tmp = input_ptr[i];
      __half2 *val_h2 = (__half2 *)(&tmp);
      float4 gamma_tmp = __ldg(reinterpret_cast<const float4 *>(gamma) + i);
      __half2 *gamma_h2 = (__half2 *)(&gamma_tmp);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 gamma_f2 = __half22float2(gamma_h2[j]);
        float2 val_f2 = __half22float2(val_h2[j]);
        val_f2.x = val_f2.x * s_variance * gamma_f2.x;
        val_f2.y = val_f2.y * s_variance * gamma_f2.y;
        val_h2[j] = __float22half2_rn(val_f2);
      }
      output_ptr[i] = tmp;
    }
}

template <>
__global__ void T5norm_kernel_opt(const nv_bfloat16 *__restrict input,
                                  const nv_bfloat16 *__restrict gamma,
                                  nv_bfloat16 *output,
                                  int m, int n)
{
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float variance = 0.0f;
    float local_var_sum = 0.0f;
    const float4 *input_ptr = reinterpret_cast<const float4 *>(input) + blockIdx.x * n;
    float4 *output_ptr = (float4 *)(output) + blockIdx.x * n;

    for (int i = tid; i < n; i += blockDim.x) {
      float4 tmp = input_ptr[i];
      nv_bfloat162 *val_h2 = (nv_bfloat162 *)(&tmp);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 val_f2 = __bfloat1622float2(val_h2[j]);
        local_var_sum += val_f2.x * val_f2.x + val_f2.y * val_f2.y;
      }
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)(n << 3) + 1e-6);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
      float4 tmp = input_ptr[i];
      nv_bfloat162 *val_h2 = (nv_bfloat162 *)(&tmp);
      float4 gamma_tmp = __ldg(reinterpret_cast<const float4 *>(gamma) + i);
      nv_bfloat162 *gamma_h2 = (nv_bfloat162 *)(&gamma_tmp);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float2 gamma_f2 = __bfloat1622float2(gamma_h2[j]);
        float2 val_f2 = __bfloat1622float2(val_h2[j]);
        val_f2.x = val_f2.x * s_variance * gamma_f2.x;
        val_f2.y = val_f2.y * s_variance * gamma_f2.y;
        val_h2[j] = __float22bfloat162_rn(val_f2);
      }
      output_ptr[i] = tmp;
    }
}

template <typename T>
void layernorm(const void *input, const void *gamma,
               const void *beta, void *output,
               const int &m, const int &n,
               const cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block;

  //n must be less than 16384
  assert(n <= 16384);
  if (n <= 1024)
  {
    block.x = ceil(n / (32.0 * 1)) * 32;         // item_per_thread = 1
    block.x = block.x / (4 / sizeof(T)); // if using half, only need half of block.x
    decoder_norm1_kernel_opt<1><<<grid, block, 0, stream>>>((T*)input, (T*)gamma, (T*)beta, (T*)output, m, n);
  }
  else if (n <= 2048)
  {
    block.x = ceil(n / (32.0 * 2)) * 32;         // item_per_thread = 2
    block.x = block.x / (4 / sizeof(T)); // if using half, only need half of block.x
    decoder_norm1_kernel_opt<2><<<grid, block, 0, stream>>>((T*)input, (T*)gamma, (T*)beta, (T*)output, m, n);
  }
  else if (n <= 4096)
  {
    block.x = ceil(n / (32.0 * 4)) * 32;         // item_per_thread = 4
    block.x = block.x / (4 / sizeof(T)); // if using half, only need half of block.x
    decoder_norm1_kernel_opt<4><<<grid, block, 0, stream>>>((T*)input, (T*)gamma, (T*)beta, (T*)output, m, n);
  }
  else if (n <= 8192)
  {
    block.x = ceil(n / (32.0 * 8)) * 32;         // item_per_thread = 8
    block.x = block.x / (4 / sizeof(T)); // if using half, only need half of block.x
    decoder_norm1_kernel_opt<8><<<grid, block, 0, stream>>>((T*)input, (T*)gamma, (T*)beta, (T*)output, m, n);
  }
  else if (n <= 16384)
  {
    block.x = ceil(n / (32.0 * 16)) * 32;        // item_per_thread = 16
    block.x = block.x / (4 / sizeof(T)); // if using half, only need half of block.x
    decoder_norm1_kernel_opt<16><<<grid, block, 0, stream>>>((T*)input,(T*)gamma, (T*)beta,(T*) output, m, n);
  }
  else
  {
    std::cout << "not support size for layernorm" << std::endl;
  }
}

template <typename T>
void T5layernorm(const void *input, const void *gamma,
                 void *output, int m, int n,
                 const cudaStream_t stream)
{
  assert(n % 8 == 0); // hidden_states % 8 == 0
  dim3 grid(m);
  dim3 block;
  if (sizeof(T) == sizeof(float)) {
    n >>= 2;
  } else {
    n >>= 3;
  }
  block.x = min(((n + 31) / 32) * 32, 1024);

  T5norm_kernel_opt<T><<<grid, block, 0, stream>>>((T*)input, (T*)gamma, (T*)output, m, n);
}

template <class T>
void add_bias_input_layernorm_kernel(void *output, const void *input,
                                     const void *bias, const void *gamma,
                                     const void *beta, const int &m, int &n,
                                     const cudaStream_t stream)
{
  if (sizeof(T) == sizeof(float))
  {
    dim3 grid(m);
    dim3 block(n);
    assert(n <= 1024);
    if (bias != nullptr) {
      if (n == 768 || n == 1024)
        add_bias_input_layernorm_v2<<<grid, n / 4, 0, stream>>>((T*)output, (T*)input, (T*)bias, (T*)gamma, (T*)beta, n);
      else
        add_bias_input_layernorm<<<grid, block, 0, stream>>>((T*)output, (T*)input, (T*)bias, (T*)gamma, (T*)beta, m, n);
    } else {
      if (n == 768 || n == 1024)
        add_input_T5layernorm_v2<<<grid, n / 4, 0, stream>>>((T*)output, (T*)input, (T*)gamma, n);
      else
        add_input_T5layernorm<<<grid, block, 0, stream>>>((T*)output, (T*)input, (T*)gamma, m, n);      
    }
  }
  else
  {
    dim3 grid(m);
    dim3 block(n / 2);
    assert(n / 2 <= 1024);

    if (bias != nullptr) {
      if (m >= 512 && (n == 768 || n == 1024))
        add_bias_input_layernorm_v2<<<grid, n / 8, 0, stream>>>((T*)output, (T*)input, (T*)bias, (T*)gamma, (T*)beta, n);
      else
        add_bias_input_layernorm<<<grid, block, 0, stream>>>((T*)output, (T*)input, (T*)bias, (T*)gamma, (T*)beta, m, n);
    } else {
      if (m >= 512 && (n == 768 || n == 1024))
        add_input_T5layernorm_v2<<<grid, n / 8, 0, stream>>>((T*)output, (T*)input, (T*)gamma, n);
      else
        add_input_T5layernorm<<<grid, block, 0, stream>>>((T*)output, (T*)input, (T*)gamma, m, n);
    }
  }
}

template void add_bias_input_layernorm_kernel<float>(void *output, const void *input,
                                                     const void *bias, const void *gamma,
                                                     const void *beta, const int &m, int &n,
                                                     const cudaStream_t stream);

template void add_bias_input_layernorm_kernel<half>(void *output, const void *input,
                                                    const void *bias, const void *gamma,
                                                    const void *beta, const int &m, int &n,
                                                    const cudaStream_t stream);

template void add_bias_input_layernorm_kernel<nv_bfloat16>(void *output, const void *input,
                                                           const void *bias, const void *gamma,
                                                           const void *beta, const int &m, int &n,
                                                           const cudaStream_t stream);

template void layernorm<float>(const void *input, const void *gamma,
                               const void *beta, void *output, const int &m, const int &n, const cudaStream_t stream);

template void layernorm<half>(const void *input, const void *gamma,
                              const void *beta, void *output, const int &m, const int &n, const cudaStream_t stream);

template void layernorm<nv_bfloat16>(const void *input, const void *gamma,
                                     const void *beta, void *output, const int &m, const int &n, const cudaStream_t stream);

template void T5layernorm<float>(const void *input, const void *gamma, void *output, int m, int n, const cudaStream_t stream);

template void T5layernorm<half>(const void *input, const void *gamma, void *output, int m, int n, const cudaStream_t stream);

template void T5layernorm<nv_bfloat16>(const void *input, const void *gamma, void *output, int m, int n, const cudaStream_t stream);

// void T5layernorm_int8(const void *input, const void *gamma, float *scale,
//                  void *output, int m, int n,
//                  const cudaStream_t stream)
// {
//   assert(n % 8 == 0); // hidden_states % 8 == 0
//   dim3 grid(m);
//   dim3 block;
//   if (sizeof(T) == sizeof(float)) {
//     n >>= 2;
//   } else {
//     n >>= 3;
//   }
//   block.x = min(((n + 31) / 32) * 32, 1024);

//   T5norm_kernel_int8<T><<<grid, block, 0, stream>>>((T*)input, (T*)gamma, (T*)output, m, n);
// }