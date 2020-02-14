/** @file main.cpp
 * Just a simple hello world using libfmt
 */
// The previous block is needed in every file for which you want to generate documentation

#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "eztrt/model.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <string>

namespace eztrt
{
cv::Mat softmax(cv::Mat in)
{
    cv::Mat res;
    cv::exp(in, res);
    auto s = cv::sum(res)[0];
    cv::multiply(res, cv::Scalar(1. / s), res);
    return res;
}

cv::Mat apply_preprocess_steps(cv::Mat in, std::string step_list)
{
    cv::Mat out = in.clone();
    for (const auto& step : step_list)
    {
        switch (step)
        {
        case 'v': // vertical flip
            cv::flip(out, out, 0);
            break;
        case 'h': // horizontal flip
            cv::flip(out, out, 1);
            break;
        case 'r': // rotate CCW 90 degrees
            cv::rotate(out, out, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        case 't': // transpose
            cv::transpose(out, out);
            break;
        case 'I': // invert intensities y = 1-x
            cv::subtract(cv::Scalar(1.0), out, out);
            break;
        case 'C': // create grayscale to color
            cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
            break;
        case 'G': // create grayscale to color
            cv::cvtColor(out, out, cv::COLOR_BGR2GRAY);
            break;
        }
    }
    return out;
}

} // namespace eztrt

template<>
struct fmt::formatter<cv::Mat> : formatter<string_view>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        auto it = ctx.begin();
        if (*it != '}')
            throw format_error("invalid format - cv::Mat formatter does not accept arguments.");
        return it;
    }

    template<typename FormatContext>
    auto format(const cv::Mat& p, FormatContext& ctx)
    {
        // ctx.out() is an output iterator to write to.
        std::ostringstream oss;
        oss << p;
        return formatter<string_view>::format(oss.str(), ctx);
    }
};

template<>
struct fmt::formatter<cv::MatExpr> : formatter<cv::Mat>
{
    template<typename FormatContext>
    auto format(const cv::MatExpr& p, FormatContext& ctx)
    {
        return formatter<cv::Mat>::format(p, ctx);
    }
};

const char* keys = "{help h usage ? |      | print this message   }"
                   "{@path          |      | path to ONNX model file   }"
                   "{@input         |      | input image to feed to the model}"
                   "{output         |      | output image path (optional) }"
                   "{batchsize bs   | 1    | batch size}"
                   "{workspace ws   | 128  | workspace size in MiB}"
                   "{preprocess     |      | preprocess string (a list of 'vhrtICG' }"
                   "{verbose v      |      | verbose output}";

int main(int argc, char* argv[])
{

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Application name v1.0.0");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string model_path  = parser.get<std::string>(0);
    std::string input_path  = parser.get<std::string>(1);
    std::string output_path = parser.get<std::string>("output");
    int         bs          = parser.get<int>("bs");
    int         ws          = parser.get<int>("ws");
    bool        verbose     = parser.has("v");
    std::string preprocess  = parser.get<std::string>("preprocess");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    eztrt::logger log("Main", verbose ? ILogger::Severity::kVERBOSE : ILogger::Severity::kINFO);
    if (model_path.empty()) { spdlog::info("No model path was set!"); }
    spdlog::info("Loading {}...", argv[1]);
    eztrt::model::params params;
    params.batchSize      = bs;
    params.workspace_size = ws * 1024 * 1024;

    eztrt::model m(params, log);

    m.load(model_path);
    if (!m.valid())
    {
        spdlog::info("Could not load network!\n{}", m.summarize());
        return 1;
    }

    spdlog::info("Loaded Network:\n{}", m.summarize());
    if (input_path.empty()) return 0;

    auto in_img = cv::imread(input_path);
    if (in_img.empty())
    {
        spdlog::error("Could not read input image {}", input_path);
        return 1;
    }

    in_img.convertTo(in_img, CV_32FC(in_img.channels()), 1. / 255.);
    if (!preprocess.empty()) in_img = eztrt::apply_preprocess_steps(in_img, preprocess);

    auto result = m.predict(in_img);
    spdlog::info("Prediction finished!");
    auto softmaxed = eztrt::softmax(result);
    spdlog::info("Final Result:\n{}", softmaxed.t());
    for (int i = 0; i < softmaxed.total(); ++i)
    {
        std::cout << i << ": ";
        for (float cnt = 0.0f; cnt < softmaxed.at<float>(i); cnt += 0.05f)
            std::cout << "*";
        std::cout << "\n";
    }

    return 0;
}
