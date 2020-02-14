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

    template<typename... ParamTypes>
    void log(Severity severity, std::string msg, ParamTypes... params)
    {
        log(severity, fmt::format(msg, params...).c_str());
    }

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

constexpr const char* to_str(nvinfer1::LayerType lt)
{
#define CASE_LT(ENUM)                                                                              \
    case nvinfer1::LayerType::k##ENUM: return (#ENUM) + 1
    switch (lt)
    {
        CASE_LT(CONVOLUTION);
        CASE_LT(FULLY_CONNECTED);
        CASE_LT(ACTIVATION);
        CASE_LT(POOLING);
        CASE_LT(LRN);
        CASE_LT(SCALE);
        CASE_LT(SOFTMAX);
        CASE_LT(DECONVOLUTION);
        CASE_LT(CONCATENATION);
        CASE_LT(ELEMENTWISE);
        CASE_LT(PLUGIN);
        CASE_LT(RNN);
        CASE_LT(UNARY);
        CASE_LT(PADDING);
        CASE_LT(SHUFFLE);
        CASE_LT(REDUCE);
        CASE_LT(TOPK);
        CASE_LT(GATHER);
        CASE_LT(MATRIX_MULTIPLY);
        CASE_LT(RAGGED_SOFTMAX);
        CASE_LT(CONSTANT);
        CASE_LT(RNN_V2);
        CASE_LT(IDENTITY);
        CASE_LT(PLUGIN_V2);
        CASE_LT(SLICE);
        CASE_LT(SHAPE);
        CASE_LT(PARAMETRIC_RELU);
        CASE_LT(RESIZE);
        CASE_LT(TRIP_LIMIT);
        CASE_LT(RECURRENCE);
        CASE_LT(ITERATOR);
        CASE_LT(LOOP_OUTPUT);
        CASE_LT(SELECT);
        CASE_LT(FILL);
    default: return "unknown";
    }
}

} // namespace eztrt
