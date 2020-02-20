#include "eztrt/util.h"
#include "eztrt/model.h"

#include "json.hpp"

namespace eztrt
{

cv::Mat softmax(cv::Mat in, int dim /*= 1*/)
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
    for_each_channel(result, [](const int& c, cv::Mat& channel) {
        cv::imshow("Output #" + std::to_string(c), channel);
    });
}

void save_all_channels(cv::Mat result, std::string file_base)
{
    for_each_channel(result, [&](const int& c, const cv::Mat& channel) {
        cv::imwrite(fmt::format(file_base, c), channel);
    });
}

std::vector<cv::Mat> separate_channels(cv::Mat result)
{
    std::vector<cv::Mat> channels;
    for_each_channel(result,
                     [&](const int& c, const cv::Mat& channel) { channels.push_back(channel); });
    return channels;
}

cv::Mat reshape_channels(cv::Mat m)
{
    std::vector<int> shape{m.size.p, m.size.p + m.dims};
    shape.push_back(m.channels());
    return m.reshape(1, shape);
}

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
    //       right now on account of tiredness and, frankly, who cares.
    m.clone().forEach<float>([&](float& v, const int* p) {
        std::vector<int> old_pos(m.dims);
        std::generate_n(begin(old_pos), m.dims, [&, i = 0]() mutable { return p[new_order[i++]]; });
        res.at<float>(old_pos.data()) = v;
    });
    return res;
}

cv::Mat try_adjust_input(cv::Mat input, int input_index, model& m)
{
    auto tensor = m.inputs()[input_index];
    auto dims   = tensor->getDimensions();
    auto type   = tensor->getType();

    if (dims.nbDims != 4)
    {
        spdlog::warn("Currently auto-adjust only works for 4-dimensional inputs");
        return {};
    }

    int W{dims.d[3]}, H{dims.d[2]}, C{dims.d[1]} /*, N{dims.d[0]}*/;
    if (dims.d[0] != 1)
    {
        spdlog::warn("We assume an internal batch size of 1 but first dimension is actually {}",
                     dims.d[0]);
    }

    // adjust size
    if (H != input.rows || W != input.cols) cv::resize(input, input, cv::Size(W, H));

    // adjust type
    double mul          = 1.0;
    double add          = 0.0;
    auto   in_elem_type = input.type() & CV_MAT_DEPTH_MASK;
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
        if (in_elem_type == CV_8U) mul = 1. / double(0xFF);
        if (in_elem_type == CV_8S) mul = 1. / double(0x7F);
        if (in_elem_type == CV_16U) mul = 1. / double(0xFFFF);
        if (in_elem_type == CV_16S) mul = 1. / double(0x7FFF);
        if (in_elem_type == CV_32S) mul = 1. / double(0x7FFFFFFF);
        input.convertTo(input, CV_32FC(input.channels()), mul, add);
        break;
    case nvinfer1::DataType::kINT32: input.convertTo(input, CV_32SC(input.channels()), mul, add);
    case nvinfer1::DataType::kINT8:
        if (in_elem_type == CV_8U) add = -double(0x7f);
        if (in_elem_type == CV_16U)
        {
            mul = double(0xFF) / double(0xFFFF);
            add = -double(0x7f);
        }
        if (in_elem_type == CV_16S) mul = double(0x7F) / double(0x7FFF);
        if (in_elem_type == CV_32S) mul = double(0xFF) / double(0x7FFFFFFF);
        if (in_elem_type == CV_32F) mul = double(0x7F);
        input.convertTo(input, CV_8SC(input.channels()), mul, add);
    default:
        spdlog::warn("Could not adjust element type - type {} not supported.", to_str(type));
        return {};
    }

    // TODO adjust input range?

    // adjust number of channels if possible
    if (C == 1 && input.channels() == 3)
        cv::cvtColor(input, input, CV_BGR2GRAY);
    else if (C == 1 && input.channels() == 4)
        cv::cvtColor(input, input, CV_BGRA2GRAY);
    else if (C == 3 && input.channels() == 1)
        cv::cvtColor(input, input, CV_GRAY2BGR);
    else if (C == 4 && input.channels() == 1)
        cv::cvtColor(input, input, CV_GRAY2BGRA);
    else if (C != input.channels())
    {
        spdlog::warn("Could not adjust number of channels to match expected input shape: "
                     "Given input has {} channels while model expects {} channels.",
                     input.channels(), C);
        return {};
    }
    // adjust channels from HWC to CHW
    if (input.dims == 2) { input = permute_dims(reshape_channels(input), {2, 0, 1}); }

    return input;
}

std::unordered_map<size_t, std::string> load_class_labels(const std::string& filename)
{
    using json = nlohmann::json;

    std::ifstream ifs(filename);
    if (ifs.is_open())
    {
        std::unordered_map<size_t, std::string> class_label_map;
        auto                                    parsed_data = json::parse(ifs);

        for (auto& [key, value] : parsed_data.items())
        {
            size_t cls_idx           = atoi(key.c_str());
            class_label_map[cls_idx] = value.get<std::string>();
        }
        return class_label_map;
    }
    return {};
}

} // namespace eztrt
