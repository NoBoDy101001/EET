#include <cublasLt.h>
#include <cublas_v2.h>
#include "op/common.hpp"

void cublasLtgemmStridedBatch(cublasLtHandle_t ltHandle,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
                              int m,
                              int n,
                              int k,
                              const void *alpha,
                              const void *A, int lda, int64_t stridea,
                              const void *B, int ldb, int64_t strideb,
                              const void *beta,
                              void *C, int ldc, int64_t stridec,
                              int batchCount,
                              cudaDataType_t dataType,
                              cudaDataType_t scaleType,
                              cublasComputeType_t computeType,
                              void *workspace,
                              size_t workspaceSize,
                              cudaStream_t stream);

void cublasLtgemm(cublasLtHandle_t ltHandle,
                  cublasOperation_t transa,
                  cublasOperation_t transb,
                  int m,
                  int n,
                  int k,
                  const void *alpha,
                  const void *A, int lda,
                  const void *B, int ldb,
                  const void *beta,
                  void *C, int ldc,
                  cudaDataType_t dataType,
                  cudaDataType_t scaleType,
                  cublasComputeType_t computeType, 
                  cublasLtMatmulAlgo_t& algo,
                  void *workspace,
                  size_t workspaceSize,
                  cudaStream_t stream,
                  void *D = nullptr);

std::pair<bool, cublasLtMatmulAlgo_t> findBestAlgo(cublasLtHandle_t ltHandle,
                                                   cublasLtMatmulDesc_t operationDesc,
                                                   const void *alpha,
                                                   const void *A, cublasLtMatrixLayout_t Adesc,
                                                   const void *B, cublasLtMatrixLayout_t Bdesc,
                                                   const void *beta,
                                                   const void *C, cublasLtMatrixLayout_t Cdesc,
                                                   void *D, cublasLtMatrixLayout_t Ddesc,
                                                   void *workspace, size_t workspaceSize,
                                                   cudaStream_t stream);

void cublasLtIgemm(cublasLtHandle_t ltHandle,
                  int m,
                  int n,
                  int k,
                  const void *alpha,
                  const int8_t *Atransform,
                  const int8_t *Btransform,
                  const void *beta,
                  int32_t *Ctransform,
                  cudaDataType_t dataType,
                  cudaDataType_t scaleType,
                  cublasComputeType_t computeType,
                  void *workspace,
                  size_t workspaceSize,
                  cudaStream_t stream);