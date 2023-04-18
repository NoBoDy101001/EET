#ifndef _OP_COMMON_HPP_
#define _OP_COMMON_HPP_

#include <stdlib.h>
#include <string>
#include <vector>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "opbase.h"

// #define _DEBUG_MODE_
// #define _AUTOTUNE_
#define QKV_PTR_SIZE 3
#define FUSED_QKV_PTR_SIZE 9
#define WORKSPACE_SIZE 32 * 1024 * 1024

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    throw std::runtime_error(std::string("[EET][ERROR] CUDA runtime error: ") + \
        (_cudaGetErrorEnum(result)) + " " + file +  \
        ":" + std::to_string(line) + " \n");\
  }
}

template void check<cudaError_t>(cudaError_t result,
                                 char const *const func,
                                 const char *const file,
                                 int const line);
template void check<cublasStatus_t>(cublasStatus_t result,
                                    char const *const func,
                                    const char *const file,
                                    int const line);

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

#define RUN_KERNEL(FUNCTION,DTYPE,...)                       \
  if (DTYPE == torch::kBFloat16) {                            \
    FUNCTION<nv_bfloat16>(__VA_ARGS__);                       \
  } else if (DTYPE == torch::kFloat32) {                      \
    FUNCTION<float>(__VA_ARGS__);                             \
  } else {                                                    \
    FUNCTION<half>(__VA_ARGS__);                              \
  }                                                           \

#define RUN_KERNEL1(FUNCTION,DTYPE,...)                        \
  if (DTYPE == torch::kFloat32) {                             \
    FUNCTION<float>(__VA_ARGS__);                             \
  } else {                                                    \
    FUNCTION<half>(__VA_ARGS__);                              \
  }                                                           \

inline void printAlgo(const cublasLtMatmulAlgo_t& algo, float time) {
    int algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption;

    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
    
    printf("algo={ Id=%d, tileIdx=%d, stages=%d, splitK=%d, reduc=%d, swizzle=%d, custom=%d }, time={ %.3f ms }\n",
        algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption, time);
}

inline int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

#endif
