#ifndef _METADESC_
#define _METADESC_

#include "op/common.hpp"
#include <unistd.h>
#include <map>
#include <fstream>

namespace eet {

// Metadata description 
class MetaDesc{
    public:
    
    MetaDesc(const int& batch_size,const int& head_num, 
                const int& hidden_units, const int& d_kv, const int& d_ff,
                const int& layer_num, 
                const int& max_seq_len,
                const int& max_full_seq_len,
                const py::object& dtype,
                const std::string& cuda_device = "cuda:0",
                const bool& requires_grad = false,
                const std::string& activation_fn = "gelu"):
            batch_size_(batch_size), 
            head_num_(head_num),
            hidden_units_(hidden_units),
            d_kv_(d_kv),
            d_ff_(d_ff),
            layer_num_(layer_num),
            max_seq_len_(max_seq_len),                      // prompt seq_len
            max_full_seq_len_(max_full_seq_len),            // max generated seq_len
            activation_fn_(activation_fn)
    {
        dtype_ = torch::python::detail::py_object_to_dtype(dtype);
    
        options_ = torch::TensorOptions().dtype(dtype_).device(cuda_device).requires_grad(requires_grad);
        switch(dtype_){
            case torch::kFloat32:
                loadAlgoMap("_fp32.cfg");
                dataType_ = CUDA_R_32F;
                scaleType_ = CUDA_R_32F;
                computeType_ = CUBLAS_COMPUTE_32F_FAST_16F; // CUBLAS_COMPUTE_32F_FAST_TF32
                break;
            case torch::kFloat16:
                loadAlgoMap("_fp16.cfg");
                dataType_ = CUDA_R_16F;
                scaleType_ = CUDA_R_16F;
                computeType_ = CUBLAS_COMPUTE_16F;
                break;
            case torch::kBFloat16:
                loadAlgoMap("_bf16.cfg");
                dataType_ = CUDA_R_16BF;
                scaleType_ = CUDA_R_32F;
                computeType_ = CUBLAS_COMPUTE_32F; // CUBLAS_COMPUTE_32F_FAST_TF32
                break;
            case torch::kInt8:
                break;
            default:
                break;
        }
        is_available(); 
        if (cublasHandle == nullptr || stream == nullptr || ltHandle == nullptr){  
            // printf("create handel\n");
            check_cuda_error(cublasLtCreate(&ltHandle));
            check_cuda_error(cublasCreate(&cublasHandle));
            check_cuda_error(cudaStreamCreate(&stream));
            check_cuda_error(cublasSetStream(cublasHandle, stream));
        }
    }

    //construct from c++
    MetaDesc(const int& batch_size,const int& head_num, const int& hidden_units,const int& layer_num,
             const int& max_seq_len,
             const int& max_full_seq_len,
             const c10::ScalarType& dtype,
             const std::string& cuda_device = "cuda:0",
             const bool& requires_grad = false,
             const std::string& activation_fn = "gelu"):
            batch_size_(batch_size),
            head_num_(head_num),
            hidden_units_(hidden_units),
            layer_num_(layer_num),
            max_seq_len_(max_seq_len),                      // prompt seq_len
            max_full_seq_len_(max_full_seq_len),            // max generated seq_len
            dtype_(dtype),
            activation_fn_(activation_fn)
    {
        options_ = torch::TensorOptions().dtype(dtype_).device(cuda_device).requires_grad(requires_grad);
        switch(dtype_){
            case torch::kFloat32:
                dataType_ = CUDA_R_32F;
                computeType_ = CUBLAS_COMPUTE_32F_FAST_16F; // CUBLAS_COMPUTE_32F_FAST_TF32
                break;
            case torch::kFloat16:
                dataType_ = CUDA_R_16F;
                computeType_ = CUBLAS_COMPUTE_16F;
                break;
            case torch::kBFloat16:
                dataType_ = CUDA_R_16BF;
                computeType_ = CUBLAS_COMPUTE_32F_FAST_16F; // CUBLAS_COMPUTE_32F_FAST_TF32
                break;
                //TODO
            case torch::kInt8:
                break;
            default:
                break;
        }
        is_available();
        if (cublasHandle == nullptr || stream == nullptr){
            // printf("create handel\n");
            check_cuda_error(cublasCreate(&cublasHandle));
            check_cuda_error(cudaStreamCreate(&stream));
            check_cuda_error(cublasSetStream(cublasHandle, stream));
        }

    }

    MetaDesc(const MetaDesc& meta) = default;
    MetaDesc& operator=(const MetaDesc& meta) = default;
    ~MetaDesc() {
        algo_map_.clear();
    }

    const int batch_size_;
    int head_num_;
    int hidden_units_;
    int d_kv_;
    int d_ff_;
    const int max_seq_len_;
    const int max_full_seq_len_;
    const int layer_num_;
    std::string activation_fn_;
    torch::TensorOptions options_;
    cudaDataType_t dataType_, scaleType_;           // cuda dtype
    cublasComputeType_t computeType_;   // cublas type
    c10::ScalarType dtype_;             // torch dtype
    std::string algo_filename;

    static std::string DEFAULT_DIR;
    static std::map<std::string, std::map<std::string, int>> algo_map_;
    static cublasHandle_t cublasHandle;
    static cublasLtHandle_t ltHandle;
    static cudaStream_t stream;


    void is_available(){
        assert(batch_size_ > 0 && "batch size must > 0");
        assert(head_num_ > 0 && "head_num must > 0");
        // assert(hidden_units_ % head_num_ == 0 && " hidden_units must a multiple of head_num");      // TODO not necessary
        assert(layer_num_ > 0 && "layer_num must > 0");
        assert(max_seq_len_ > 0 && "max_seq_len must > 0");
        assert(max_full_seq_len_ > 0 && "max_seq_len must > 0");
        assert((options_.dtype() == torch::kFloat32 || options_.dtype() == torch::kFloat16 || options_.dtype() == torch::kBFloat16 ||
        options_.dtype() == torch::kInt8) && "EET now only support float / half / bfloat16 / int8" );
        assert(options_.device().is_cuda() && "EET now only support CUDA");
        assert(options_.requires_grad() == false && "EET now only support inference");
    }

    std::string getGPUName() {
        int device{-1};
        check_cuda_error(cudaGetDevice(&device));
        cudaDeviceProp props;
        check_cuda_error(cudaGetDeviceProperties(&props, device));
        std::string full_name = std::string(props.name);
        std::vector<std::string> name_list = {"3090", "A30"};
        for (auto name : name_list) {
            if (full_name.find(name) != std::string::npos) {
            return name;
            }
        }
        return "";
    }


    void loadAlgoMap(std::string suffix) {
        algo_filename = DEFAULT_DIR + getGPUName() + suffix;

        FILE* fp;
        fp = fopen(algo_filename.c_str(), "r");
        if (fp == NULL) {
            std::cout << "[EET][WARNING] " << algo_filename << " is not found, using default config" << std::endl;
            return;
        }
        std::cout << "Get GEMM config from " << algo_filename << std::endl;
        int m, n, k, algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption;
        while (fscanf(fp, "%d %d %d %d %d %d %d %d %d %d", &m, &n, &k, &algoId, &tile, &stages, &numSplitsK, &reductionScheme, &swizzle, &customOption) == 10) {
            char mnk[32];
            sprintf(mnk, "%d_%d_%d", m, n, k);
            if (algo_map_.find(mnk) == algo_map_.end()) {
                algo_map_[mnk]["algoId"] = algoId;
                algo_map_[mnk]["tile"] = tile;
                algo_map_[mnk]["stages"] = stages;
                algo_map_[mnk]["numSplitsK"] = numSplitsK;
                algo_map_[mnk]["reductionScheme"] = reductionScheme;
                algo_map_[mnk]["swizzle"] = swizzle;
                algo_map_[mnk]["customOption"] = customOption;   
            }
        }
        fclose(fp);
    }

    std::pair<bool, cublasLtMatmulAlgo_t> getAlgo(int m, int n, int k) {
        cublasLtMatmulAlgo_t algo;
        char mnk[32];
        sprintf(mnk, "%d_%d_%d", m, n, k);
        if (algo_map_.find(mnk) != algo_map_.end()) {
            int algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption;
            algoId = algo_map_[mnk]["algoId"];
            tile = algo_map_[mnk]["tile"];
            stages = algo_map_[mnk]["stages"];
            numSplitsK = algo_map_[mnk]["numSplitsK"];
            reductionScheme = algo_map_[mnk]["reductionScheme"];
            swizzle = algo_map_[mnk]["swizzle"];
            customOption = algo_map_[mnk]["customOption"];

            cublasLtMatmulAlgoInit(ltHandle, computeType_, scaleType_, dataType_, dataType_, dataType_, dataType_, algoId, &algo);
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(numSplitsK), sizeof(numSplitsK));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(reductionScheme));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));  
            return {true, algo};
        } else {
            int tile = -1;
            cublasLtMatmulAlgoInit(ltHandle, computeType_, scaleType_, dataType_, dataType_, dataType_, dataType_, -1, &algo);
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
            return {false, algo};
        }
    }

    void saveAlgo(int m, int n, int k, const cublasLtMatmulAlgo_t& algo) {
        char mnk[32];
        sprintf(mnk, "%d_%d_%d", m, n, k);
        if (algo_map_.find(mnk) == algo_map_.end()) {
            int algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption;

            cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
            cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
            cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
            cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
            cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
            cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
            cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
            algo_map_[mnk]["algoId"] = algoId;
            algo_map_[mnk]["tile"] = tile;
            algo_map_[mnk]["stages"] = stages;
            algo_map_[mnk]["numSplitsK"] = numSplitsK;
            algo_map_[mnk]["reductionScheme"] = reductionScheme;
            algo_map_[mnk]["swizzle"] = swizzle;
            algo_map_[mnk]["customOption"] = customOption;

            FILE* fp;
            fp = fopen(algo_filename.c_str(), "a+");
            if (fp == NULL) {
                std::cout << "[EET][WARNING] " << algo_filename << " is not found, can not save gemm config" << std::endl;
                return;
            }
            fprintf(fp, "%d %d %d %d %d %d %d %d %d %d\n", m, n, k, algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption);
            fclose(fp);
        } else {
            return;
        }
    }

};  // class MetaDesc
}   // namespace eet

#endif