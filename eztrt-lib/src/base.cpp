#include "eztrt/base.h"

#include <spdlog/spdlog.h>

#include <numeric>

namespace eztrt
{

logger::logger(std::string cat, nvinfer1::ILogger::Severity level) : level_{level}
{
    contexts_.push_back(cat);
}

void logger::log(nvinfer1::ILogger::Severity severity, const char* msg)
{
    // suppress lower-level messages
    if (static_cast<int>(severity) > static_cast<int>(level_)) return;

    auto cat = std::accumulate(next(begin(contexts_)), end(contexts_), contexts_[0],
                               [](const auto& acc, const auto& val) { return acc + ">" + val; });

    switch (severity)
    {
    case ILogger::Severity::kVERBOSE: spdlog::debug("{} {}", cat, msg); break;
    case ILogger::Severity::kINFO: spdlog::info("{} {}", cat, msg); break;
    case ILogger::Severity::kWARNING: spdlog::warn("{} {}", cat, msg); break;
    case ILogger::Severity::kERROR: spdlog::error("{} {}", cat, msg); break;
    case ILogger::Severity::kINTERNAL_ERROR:
        spdlog::error("{}[INTERNAL ERROR] {}", cat, msg);
        break;
    }
}

} // namespace eztrt
