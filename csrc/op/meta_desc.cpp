#include "op/meta_desc.hpp"
namespace eet{
    cublasHandle_t MetaDesc::cublasHandle = nullptr;
    cublasLtHandle_t MetaDesc::ltHandle = nullptr;
    cudaStream_t MetaDesc::stream = nullptr;
    std::map<std::string, std::map<std::string, int>> MetaDesc::algo_map_;
    std::string MetaDesc::DEFAULT_DIR = std::string(std::getenv("EET_HOME")) + "/example/python/resource/eet_";
}
