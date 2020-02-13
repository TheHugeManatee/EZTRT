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
        if (m.valid())
        {
            spdlog::info("Loaded Network:\n{}", m.summarize());
            if (argc > 2)
            {
                std::string fn     = argv[2];
                auto        in_img = cv::imread(fn);
                if (in_img.empty())
                {
                    spdlog::error("Could not read input image {}", fn);
                    return 1;
                }

                cv::cvtColor(in_img, in_img, cv::COLOR_BGR2GRAY);
                in_img.convertTo(in_img, CV_32FC1, 1. / 255.);
                in_img = 1 - in_img;

                auto result = m.predict(in_img);
                spdlog::info("Prediction finished!");
                std::cout << result;
            }
        }

        else
        {
            spdlog::info("Could not load network!\n{}", m.summarize());
        }
    }
    return 0;
}
