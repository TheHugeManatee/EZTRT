#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "eztrt/model.h"
#include "eztrt/util.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

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

const char* keys =
    "{help h usage ? |      | print this message   }"
    "{@path          |      | path to ONNX model file   }"
    "{@input         |      | input image or video to feed to the model. CAMERA0 and CAMERA1 are "
    "special sources that use an opencv video capture to query a webcam.}"
    "{output         |      | output image path (optional). Leave a pair of curly "
    "braces in there to output per-channel}"
    "{engine         |      | file to a serialized engine blob}"
    "{bs             | 1    | batch size}"
    "{ws             | 128  | workspace size in MiB}"
    "{preprocess     |      | preprocess string as a list/subset of v,h,r,t,I,C,G }"
    "{v              |      | verbose output}";

int main(int argc, char* argv[])
{
    using namespace eztrt;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Application name v1.0.0");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string model_path         = parser.get<std::string>(0);
    std::string input_path         = parser.get<std::string>(1);
    std::string output_path        = parser.get<std::string>("output");
    int         bs                 = parser.get<int>("bs");
    int         ws                 = parser.get<int>("ws");
    bool        verbose            = parser.has("v");
    std::string preprocess         = parser.get<std::string>("preprocess");
    std::string engine_path        = parser.get<std::string>("engine");
    bool        engine_path_exists = file_exists(engine_path);
    int         camera             = input_path == "CAMERA0" ? 0 : input_path == "CAMERA1" ? 1 : -1;

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    logger log("Main", ILogger::Severity::kVERBOSE);
    spdlog::set_level(verbose ? spdlog::level::debug : spdlog::level::info);
    if (verbose) spdlog::info("Verbose mode enabled.");
    if (model_path.empty())
    {
        spdlog::info("No model path was set!");
        return 1;
    }

    model::params params;
    params.batchSize      = bs;
    params.workspace_size = ws * 1024 * 1024;

    model m(params, log);

    spdlog::info("Loading {}...", model_path);
    m.load(model_path, engine_path_exists ? engine_path : "");
    if (!m.ready())
    {
        spdlog::info("Could not load network!\n{}", m.summarize());
        return 1;
    }

    // if engine file didn't exist before, serialize the created engine
    if (!engine_path_exists && !engine_path.empty()) m.serialize_engine(engine_path);

    spdlog::info("Loaded Network:\n{}", m.summarize());

    cv::VideoCapture src;

    if (camera >= 0)
        src = cv::VideoCapture(0);
    else
        src = cv::VideoCapture(input_path);

    cv::Mat in_img, in_data;

    while (src.read(in_img))
    {
        if (!preprocess.empty())
            in_data = apply_preprocess_steps(in_img, preprocess);
        else
            in_data = in_img;

        in_data = try_adjust_input(in_data, 0, m);

        auto result = m.predict(in_data);
        spdlog::info("Prediction finished!");

        auto softmaxed = softmax(result);

        if (softmaxed.dims == 4)
        {
            show_all_channels(softmaxed);
            cv::waitKey(-1);
            if (!output_path.empty()) save_all_channels(softmaxed, output_path);
        }
        else
        {
            //  display a 1D vector (classification) as a softmaxed "bar" chart
            spdlog::info("Final Result after softmax, classes with p>0.05:");
            for (int i = 0; i < softmaxed.total(); ++i)
            {
                auto prob = softmaxed.at<float>(i);
                if (prob < 0.05) continue;
                std::string stars;
                for (float cnt = 0.0f; cnt < 1.0f; cnt += 1.f / 19.f)
                    stars += cnt > prob ? " " : "*";
                spdlog::info("{:3}: {} [{}%]", i, stars, floorf(prob * 1000.f) / 10.f);
            }
        }

        cv::imshow("input", in_img);
        int key = cv::waitKey(-1);
        if ((key & 0xFF) == 27) break;
        src.read(in_img);
    }
    spdlog::info("No more data to process!");
    return 0;
}
