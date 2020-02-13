#pragma once

#include "NvInfer.h"

#include <cstdint>
#include <memory>
#include <stack>
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
    struct context_holder
    {
        context_holder(std::string cat, logger& logger) : logger_{logger}
        {
            logger_.contexts_.push_back(cat);
        }
        ~context_holder() { logger_.contexts_.pop_back(); }

    private:
        logger& logger_;
    };

public:
    context_holder context_scope(std::string cat) { return context_holder(cat, *this); }

    logger(std::string cat, Severity level = ILogger::Severity::kINFO);
    void log(Severity severity, const char* msg) override;

private:
    Severity                 level_;
    std::vector<std::string> contexts_;
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
