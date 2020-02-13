#include "eztrt/base.h"

#include <spdlog/spdlog.h>

namespace eztrt
{

logger::logger(std::string cat, nvinfer1::ILogger::Severity level) : cat_{cat}, level_{level} {}

void logger::log(nvinfer1::ILogger::Severity severity, const char* msg)
{
    // suppress lower-level messages
    if (static_cast<int>(severity) >= static_cast<int>(level_)) return;

    if (severity == Severity::kINFO) spdlog::info("{}: {}", cat_, msg);
    if (severity == Severity::kWARNING) spdlog::warn("{}: {}", cat_, msg);
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)
        spdlog::error("{}: {}", cat_, msg);
}

} // namespace eztrt
