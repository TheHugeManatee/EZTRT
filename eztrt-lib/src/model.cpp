
#include "eztrt/model.h"

namespace eztrt
{

model::model(params params, logger& logger) : params_{params}, logger_{logger}
{
    auto logctx_ = logger_.context_scope("construct");
    builder_     = eztrt::InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
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

bool model::predict(cv::Mat input)
{
    auto logctx_ = logger_.context_scope("predict");
    assert(engine_ && network_ && config_);

    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(engine_, params_.batchSize);

    if (!context_)
        context_ = InferUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_)
    {
        logger_.log(ILogger::Severity::kERROR, "Could not create an execution context!");
        return false;
    }

    // Read the input data into the managed buffers
    assert(params_.inputTensorNames.size() == 1);
    // if (!processInput(buffers)) { return false; }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) { return false; }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    // if (!verifyOutput(buffers)) { return false; }

    return true;
}

std::string model::summarize()
{
    std::stringstream summary;

    if (!network_)
        summary << "!!! No Network loaded!\n";
    else
    {
        summary << " ** Network:\n";
        auto NumInputs = network_->getNbInputs();

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
    }

    return summary.str();
}

bool model::create_engine()
{
    auto logctx_ = logger_.context_scope("create_engine");
    assert(network_ && config_);
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder_->buildEngineWithConfig(*network_, *config_), InferDeleter());

    if (!engine_)
    {
        logger_.log(ILogger::Severity::kERROR, "Could not create engine!");
        return false;
    }
    return true;
}

bool model::valid() { return builder_ && network_ && config_ && engine_ && context_; }

bool model::load(std::string file)
{
    auto logctx_ = logger_.context_scope("load");
    logger_.log(ILogger::Severity::kVERBOSE, "Created Model");
    auto parser = eztrt::InferUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(network(), logger_));
    if (!parser) { return false; }
    logger_.log(ILogger::Severity::kVERBOSE, "Created ONNX Parser");

    if (!parser->parseFromFile(file.c_str(), 1))
    {
        logger_.log(ILogger::Severity::kERROR,
                    fmt::format("Could not successfully parse {}", file).c_str());
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            auto err = parser->getError(i);
            logger_.log(ILogger::Severity::kERROR,
                        fmt::format("[{}] {}", err->code(), err->desc()).c_str());
        }
        return false;
    }

    apply_params();

    create_engine();

    return true;
}

void model::apply_params()
{
    auto logctx_ = logger_.context_scope("apply_params");
    builder().setMaxBatchSize(params_.batchSize);
    if (params_.workspace_size) config().setMaxWorkspaceSize(params_.workspace_size);

    if (params_.fp16)
    {
        config().setFlag(BuilderFlag::kFP16);
        logger_.log(ILogger::Severity::kVERBOSE, "Enabled FP16 Mode");
    }
    if (params_.int8)
    {
        config().setFlag(BuilderFlag::kINT8);
        // TODO parametrize this!
        samplesCommon::setAllTensorScales(&network(), 127.0f, 127.0f);
        logger_.log(ILogger::Severity::kVERBOSE, "Enabled INT8 Mode");
    }

    samplesCommon::enableDLA(&builder(), &config(), params_.dlaCore);
}

} // namespace eztrt
