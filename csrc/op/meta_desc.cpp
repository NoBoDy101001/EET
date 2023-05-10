#include "op/meta_desc.hpp"
namespace eet{
    cublasHandle_t MetaDesc::cublasHandle = nullptr;
    cublasLtHandle_t MetaDesc::ltHandle = nullptr;
    cudaStream_t MetaDesc::stream = nullptr;
    std::map<std::string, std::map<std::string, int>> MetaDesc::algo_map_;
}
