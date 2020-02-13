/** @file main.cpp
 * Just a simple hello world using libfmt
 */
// The previous block is needed in every file for which you want to generate documentation

#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "eztrt/bar.h"

#include <string>

namespace eztrt
{
constexpr const char* to_str(nvinfer1::DataType dt)
{
    switch (dt)
    {
    case DataType::kBOOL: return "bool";
    case DataType::kFLOAT: return "float";
    case DataType::kHALF: return "half";
    case DataType::kINT32: return "int32";
    case DataType::kINT8: return "int8";
    default: return "unknown";
    }
}

class model
{
public:
    model(ILogger& logger) : logger_{logger}
    {
        builder_ = eztrt::InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
        if (!builder_)
        {
            logger_.log(ILogger::Severity::kERROR,
                        fmt::format("Could not instantiate builder!").c_str());
            return;
        }

        const auto explicitBatch =
            1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network_ = eztrt::InferUniquePtr<nvinfer1::INetworkDefinition>(
            builder_->createNetworkV2(explicitBatch));
        if (!network_)
        {
            logger_.log(ILogger::Severity::kERROR,
                        fmt::format("Could not instantiate network instance!").c_str());
            return;
        }

        config_ = eztrt::InferUniquePtr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
        if (!config_)
        {
            logger_.log(ILogger::Severity::kERROR,
                        fmt::format("Could not instantiate builder config!").c_str());
            return;
        }
    }

    // Non-copyable
    model(const model& rhs) = delete;
    model& operator=(const model& rhs) = delete;
    // Movable
    model(model&& rhs) = default;
    model& operator=(model&& rhs) = default;

    ~model() = default;

    std::string summarize()
    {
        auto NumInputs = network_->getNbInputs();

        std::stringstream summary;
        for (int i = 0; i < NumInputs; ++i)
        {
            auto               input = network_->getInput(i);
            std::ostringstream dims;
            dims << input->getDimensions();
            summary << fmt::format("Input #{}: [{}] {} {}\n", i, input->getName(), dims.str(),
                                   to_str(input->getType()));
        }

        auto NumOutputs = network_->getNbOutputs();
        for (int i = 0; i < NumOutputs; ++i)
        {
            auto               output = network_->getOutput(i);
            std::ostringstream dims;
            dims << output->getDimensions();
            summary << fmt::format("Output #{}: [{}] {} {}\n", i, output->getName(), dims.str(),
                                   to_str(output->getType()));
        }
        return summary.str();
    }

    bool valid() { return builder_ && network_; }

    nvinfer1::INetworkDefinition& network() { return *network_; }
    nvinfer1::IBuilder&           builder() { return *builder_; }
    nvinfer1::IBuilderConfig&     config() { return *config_; }

private:
    eztrt::InferUniquePtr<nvinfer1::IBuilder>           builder_;
    eztrt::InferUniquePtr<nvinfer1::INetworkDefinition> network_;
    eztrt::InferUniquePtr<nvinfer1::IBuilderConfig>     config_;

    ILogger& logger_;
};

std::unique_ptr<model> parse_file(std::string file, SampleParams params, logger log)
{
    std::unique_ptr<model> m = make_unique<model>(log);
    log.log(ILogger::Severity::kVERBOSE, "Created Model");
    auto parser =
        eztrt::InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(m->network(), log));
    if (!parser) { return {}; }
    log.log(ILogger::Severity::kVERBOSE, "Created ONNX Parser");

    if (!parser->parseFromFile(file.c_str(), 1))
    {
        spdlog::error("Could not successfully parse {}", file);
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            auto err = parser->getError(i);
            spdlog::error("[{}] {}", err->code(), err->desc());
        }
        return {};
    }

    m->builder().setMaxBatchSize(params.batchSize);
    if (params.workspace_size) m->config().setMaxWorkspaceSize(params.workspace_size);

    if (params.fp16) { m->config().setFlag(BuilderFlag::kFP16); }
    if (params.int8)
    {
        m->config().setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(&m->network(), 127.0f, 127.0f); // TODO parametrize this!
    }

    samplesCommon::enableDLA(&m->builder(), &m->config(), params.dlaCore);

    return m;
}

} // namespace eztrt

int main(int argc, char* argv[])
{
    eztrt::logger log("Main", ILogger::Severity::kINTERNAL_ERROR);

    if (argc > 1)
    {
        spdlog::info("Loading {}...", argv[1]);
        eztrt::SampleParams params;
        params.batchSize = 8;

        auto m = eztrt::parse_file(argv[1], params, log);
        if (m->valid()) { spdlog::info("Loaded Network: {}", m->summarize()); }
        else
        {
            spdlog::info("Could not load network!");
        }
    }
    return 0;
}
