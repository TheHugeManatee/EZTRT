#pragma once

#include "NvInfer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eztrt
{
struct InferDeleter
{
    template<typename T>
    void operator()(T* obj) const
    {
        if (obj) { obj->destroy(); }
    }
};
template<typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

class logger : public nvinfer1::ILogger
{
public:
    logger(std::string cat, Severity level = ILogger::Severity::kINFO);
    void log(Severity severity, const char* msg) override;

private:
    std::string cat_;
    Severity    level_;
};

constexpr const char* to_str(nvinfer1::DataType dt)
{
    switch (dt)
    {
    case nvinfer1::DataType::kBOOL: return "bool";
    case nvinfer1::DataType::kFLOAT: return "float";
    case nvinfer1::DataType::kHALF: return "half";
    case nvinfer1::DataType::kINT32: return "int32";
    case nvinfer1::DataType::kINT8: return "int8";
    default: return "unknown";
    }
}

} // namespace eztrt
