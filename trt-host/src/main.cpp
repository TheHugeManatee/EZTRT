/** @file main.cpp
 * Just a simple hello world using libfmt
 */
// The previous block is needed in every file for which you want to generate documentation

#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "eztrt/model.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

namespace eztrt
{
cv::Mat softmax(cv::Mat in, int dim = 1)
{
    cv::Mat res;
    cv::exp(in, res);
    if (in.dims == 2 && in.size[0] == 1)
    {
        auto s = cv::sum(res)[0];
        cv::multiply(res, cv::Scalar(1. / s), res);
    }
    else
    {
        assert(res.type() == CV_32FC1 && "Generic softmax not implemented for non-float arrays");
        // we abuse the sub-range and foreach iteration to do a concise iteration scheme over the
        // dimension we need
        std::vector<cv::Range> projectedRange(res.dims, cv::Range::all());
        projectedRange[dim] = cv::Range(0, 1);
        res(projectedRange).forEach<float>([&](float& v, const int* p) {
            std::vector<int> pos(p, p + res.dims);

            // first pass - sum up
            float sum{0.0f};
            for (pos[dim] = 0; pos[dim] < res.size[dim]; ++pos[dim])
                sum += res.at<float>(pos.data());

            // second pass - divide
            for (pos[dim] = 0; pos[dim] < res.size[dim]; ++pos[dim])
                res.at<float>(pos.data()) /= sum;
        });
    }
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
        case 'S': //
            break;
        }
    }
    return out;
}

void show_all_channels(cv::Mat result)
{
    assert(result.dims == 4 &&
           "This method only works for 4-dimensional tensors of shape (1, C, H, W)");

    int nChannels = result.size[1];
    for (int c{}; c < nChannels; ++c)
    {
        std::vector<cv::Range> ranges{cv::Range(0, 1), cv::Range(c, c + 1), cv::Range::all(),
                                      cv::Range::all()};
        auto                   channel = result(ranges).reshape(1, result.size[2]);
        cv::imshow("Output #" + std::to_string(c), channel);
    }
    cv::waitKey(-1);
}

/**
 * Reshapes a Mat to make the channels an explicit dimension. I.e. a
 * 3-channel image of (128,64) size will result in a (128,64,3) 1-channel image
 */
cv::Mat reshape_channels(cv::Mat m)
{
    std::vector<int> shape{m.size.p, m.size.p + m.dims};
    shape.push_back(m.channels());
    return m.reshape(1, shape);
}

/**
 * Reorders the dimensions in-place and returns a new cv::Mat header with the appropriate shape.
 * this is equivalent to
 */
cv::Mat permute_dims(cv::Mat m, std::vector<int> new_order)
{
    assert(new_order.size() == m.dims && "New dimension order length must match matrix dimensions");
    assert(m.type() == CV_32FC1 && "Generic softmax not implemented for non-float arrays");

    std::vector<int> new_shape(m.dims);
    std::generate_n(begin(new_shape), m.dims,
                    [&, i = 0]() mutable { return m.size[new_order[i++]]; });
    auto res = m.reshape(m.channels(), m.dims, new_shape.data());

    // \TODO: this is pretty un-performant. I do a copy so I don't have to worry about in-place
    //       swapping, which would be way more efficient. however, can't figure out a good policy
    //       right now on account of tiredness..
    m.clone().forEach<float>([&](float& v, const int* p) {
        std::vector<int> old_pos(m.dims);
        std::generate_n(begin(old_pos), m.dims, [&, i = 0]() mutable { return p[new_order[i++]]; });
        res.at<float>(old_pos.data()) = v;
    });
    return res;
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
    using namespace eztrt;
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

    logger log("Main", verbose ? ILogger::Severity::kVERBOSE : ILogger::Severity::kINFO);
    if (model_path.empty()) { spdlog::info("No model path was set!"); }
    spdlog::info("Loading {}...", argv[1]);
    model::params params;
    params.batchSize      = bs;
    params.workspace_size = ws * 1024 * 1024;

    model m(params, log);

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
    if (!preprocess.empty()) in_img = apply_preprocess_steps(in_img, preprocess);

    in_img = permute_dims(reshape_channels(in_img), {2, 0, 1});

    auto result = m.predict(in_img);
    spdlog::info("Prediction finished!");

    auto softmaxed = softmax(result);
    show_all_channels(softmaxed);

    //  display a 1D vector (classification) as a softmaxed "bar" chart
    //     auto softmaxed = softmax(result);
    //     spdlog::info("Final Result:\n{}", softmaxed.t());
    //     for (int i = 0; i < softmaxed.total(); ++i)
    //     {
    //         std::cout << i << ": ";
    //         for (float cnt = 0.0f; cnt < softmaxed.at<float>(i); cnt += 0.05f)
    //             std::cout << "*";
    //         std::cout << "\n";
    //     }

    return 0;
}
