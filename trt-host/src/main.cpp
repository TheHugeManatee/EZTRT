/** @file main.cpp
 * Just a simple hello world using libfmt
 */
// The previous block is needed in every file for which you want to generate documentation

#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "eztrt/model.h"

#include <opencv2/opencv.hpp>
#include <string>

namespace eztrt
{

} // namespace eztrt

int main(int argc, char* argv[])
{
    eztrt::logger log("Main", ILogger::Severity::kVERBOSE);

    if (argc > 1)
    {
        spdlog::info("Loading {}...", argv[1]);
        eztrt::model::params params;
        params.batchSize = 8;

        eztrt::model m(params, log);

        m.load(argv[1]);
        if (m.valid()) { spdlog::info("Loaded Network:\n{}", m.summarize()); }
        else
        {
            spdlog::info("Could not load network!");
            m.summarize();
        }
    }
    return 0;
}
