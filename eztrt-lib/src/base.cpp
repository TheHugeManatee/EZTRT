#include "eztrt/base.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace eztrt
{

logger::logger(std::string cat, nvinfer1::ILogger::Severity level) : level_{level}
{
    contexts_.push_back(cat);
}

void logger::log(nvinfer1::ILogger::Severity severity, const char* msg)
{

    // suppress lower-level messages
    if (static_cast<int>(severity) >= static_cast<int>(level_)) return;

    //     auto cat = std::accumulate(
    //         next(begin(contexts_)), end(contexts_), "",
    //         [](const std::string& acc, const std::string& val) { return acc + ">" + val; });

    std::string cat = "";
    for (const auto& c : contexts_)
        cat += c + ">";

    if (severity == Severity::kINFO) spdlog::info("{} {}", cat, msg);
    if (severity == Severity::kWARNING) spdlog::warn("{} {}", cat, msg);
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)
        spdlog::error("{} {}", cat, msg);
}

} // namespace eztrt
