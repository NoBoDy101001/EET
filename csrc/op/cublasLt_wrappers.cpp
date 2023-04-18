#include "cublasLt_wrappers.hpp"

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
                              cudaStream_t stream)
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    
    cublasLtMatmulDescCreate(&operationDesc, computeType, dataType);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    
    // create matrix descriptors, we need to configure batch size and counts in this case
    cublasLtMatrixLayoutCreate(&Adesc, dataType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, dataType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, dataType, m, n, ldc);

    if (batchCount > 1) {
        cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
        cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
        cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
        cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
        cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
        cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));
    }

    check_cuda_error(cublasLtMatmul(ltHandle,
                                    operationDesc,
                                    alpha,
                                    A, Adesc,
                                    B, Bdesc,
                                    beta,
                                    C, Cdesc,
                                    C, Cdesc,
                                    NULL,
                                    workspace,
                                    workspaceSize,
                                    stream));
    
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtMatmulDescDestroy(operationDesc);    
}

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
                  void *D)
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    D = (D == nullptr) ? C : D;
    cublasLtMatmulDescCreate(&operationDesc, computeType, dataType);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    // create matrix descriptors
    cublasLtMatrixLayoutCreate(&Adesc, dataType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, dataType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);

    cublasLtMatrixLayoutCreate(&Cdesc, dataType, m, n, ldc);
#ifdef _AUTOTUNE_
    int tile;
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    if (tile == -1) {
        std::cout << "m: " << m << " n: " << n << " k: " << k << std::endl;
        auto result = findBestAlgo(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, workspace, workspaceSize, stream);
        algo = result.second;
    }
#endif

    check_cuda_error(cublasLtMatmul(ltHandle,
                                    operationDesc,
                                    alpha,
                                    A, Adesc,
                                    B, Bdesc,
                                    beta,
                                    D, Cdesc,
                                    C, Cdesc,
                                    &algo,
                                    workspace,
                                    workspaceSize,
                                    stream));
    
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtMatmulDescDestroy(operationDesc);
}

std::pair<bool, cublasLtMatmulAlgo_t> findBestAlgo(cublasLtHandle_t ltHandle,
                                                   cublasLtMatmulDesc_t operationDesc,
                                                   const void *alpha,
                                                   const void *A, cublasLtMatrixLayout_t Adesc,
                                                   const void *B, cublasLtMatrixLayout_t Bdesc,
                                                   const void *beta,
                                                   const void *C, cublasLtMatrixLayout_t Cdesc,
                                                   void *D, cublasLtMatrixLayout_t Ddesc,
                                                   void *workspace, size_t workspaceSize,
                                                   cudaStream_t stream)
{
    size_t returnSize;
    int32_t pointer_mode;
    cublasLtMatmulDescGetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode), &returnSize);

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(200);
    cublasLtMatmulPreference_t preference;
    check_cuda_error(cublasLtMatmulPreferenceCreate(&preference));
    check_cuda_error(cublasLtMatmulPreferenceInit(preference));
    uint32_t pointer_mode_mask = 0;
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &pointer_mode_mask, sizeof(pointer_mode_mask)));

    int return_count = 0;
    auto ret = cublasLtMatmulAlgoGetHeuristic(ltHandle,
                                              operationDesc,
                                              Adesc,
                                              Bdesc,
                                              Cdesc,
                                              Ddesc,
                                              preference,
                                              heuristics.size(),
                                              heuristics.data(),
                                              &return_count);
    heuristics.resize(return_count);

    std::map<int, std::vector<float>> algo_results;
    for (int searchIdx = 0; searchIdx < return_count + 1; searchIdx++) {
        cublasLtMatmulAlgo_t algo = heuristics[searchIdx].algo;
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        for (int i = 0; i < 11; i++) {
            float duration_ms;
            cudaEventRecord(start_event, stream);
            check_cuda_error(cublasLtMatmul(ltHandle,
                                            operationDesc,
                                            alpha,
                                            A,
                                            Adesc,
                                            B,
                                            Bdesc,
                                            beta,
                                            C,
                                            Cdesc,
                                            D,
                                            Ddesc,
                                            &algo,
                                            workspace,
                                            workspaceSize,
                                            stream));
            cudaEventRecord(stop_event, stream);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&duration_ms, start_event, stop_event);

            algo_results[searchIdx].push_back(duration_ms);
        }
        std::sort(algo_results[searchIdx].begin(), algo_results[searchIdx].end());
        printAlgo(algo, algo_results[searchIdx][5]);
    }

    cublasLtMatmulHeuristicResult_t result;
    float best_time = INFINITY;
    int best_idx;
    for (int searchIdx = 0; searchIdx < return_count + 1; searchIdx++) {
        const auto& results = algo_results[searchIdx];
        if (results.size() > 0 && results[5] < best_time) {
            best_time = results[5];
            best_idx = searchIdx;
        }
    }
    result = heuristics[best_idx];
    return {best_time != INFINITY, result.algo};    
}


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
                  cudaStream_t stream)
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    // int32_t *Ctransform = NULL;

    // tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
    cublasOperation_t opTranspose     = CUBLAS_OP_T;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int ldatransform = 32 * m;
    int ldbtransform = 32 * roundoff(n, 8);
    int ldctransform = 32 * m;

    cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F);
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose));
    // create matrix descriptors
    cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldatransform);
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, k, n, ldbtransform);
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C));
    cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldctransform);
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));

    check_cuda_error(cublasLtMatmul(ltHandle,
                                    operationDesc,
                                    alpha,
                                    Atransform, AtransformDesc,
                                    Btransform, BtransformDesc,
                                    beta,
                                    Ctransform, CtransformDesc,
                                    Ctransform, CtransformDesc,
                                    NULL,
                                    workspace,
                                    workspaceSize,
                                    stream));
    
        cublasLtMatrixLayoutDestroy(AtransformDesc);
        cublasLtMatrixLayoutDestroy(BtransformDesc);
        cublasLtMatrixLayoutDestroy(CtransformDesc);
        cublasLtMatmulDescDestroy(operationDesc);    
        cublasLtMatrixTransformDescDestroy(transformDesc);
}