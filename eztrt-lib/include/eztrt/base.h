#pragma once

#include "NvInfer.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <string_view>

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

protected:
    std::string prefix();

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
    case nvinfer1::LayerType::k##ENUM: return (#ENUM)
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
#undef CASE_LT
}

inline bool file_exists(std::string filename)
{
    std::ifstream f(filename);
    return f.good();
}

// from https://www.bfilipek.com/2018/07/string-view-perf-followup.html
inline std::vector<std::string_view> split(std::string_view str, std::string_view delims = " ")
{
    std::vector<std::string_view> output;

    for (auto first = str.data(), second = str.data(), last = first + str.size();
         second != last && first != last; first = second + 1)
    {
        second = std::find_first_of(first, last, std::cbegin(delims), std::cend(delims));

        if (first != second) output.emplace_back(first, second - first);
    }

    return output;
}

} // namespace eztrt
