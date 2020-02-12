#include "eztrt/bar.h"

#include <iostream>

#include "spdlog/spdlog.h"

namespace eztrt
{
void Logger::log(Severity severity, const char* msg)
{
    // suppress info-level messages
    if (severity != Severity::kINFO) spdlog::info("{}: {}", cat_, msg);
}

bool SampleOnnxMNIST::build()
{
    auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(mLogger));
    if (!builder) { return false; }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network =
        InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) { return false; }

    auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) { return false; }

    auto parser =
        InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
    if (!parser) { return false; }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed) { return false; }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), InferDeleter());
    if (!mEngine) { return false; }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(InferUniquePtr<nvinfer1::IBuilder>&           builder,
                                       InferUniquePtr<nvinfer1::INetworkDefinition>& network,
                                       InferUniquePtr<nvinfer1::IBuilderConfig>&     config,
                                       InferUniquePtr<nvonnxparser::IParser>&        parser)
{
    auto parsed = parser->parseFromFile((mParams.dataDirs[0] + "/model.onnx").c_str(),
                                        static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    if (!parsed) { return false; }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16 * 1024 * 1024);
    if (mParams.fp16) { config->setFlag(BuilderFlag::kFP16); }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer(uint8_t* /*data*/)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = InferUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context) { return false; }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers)) { return false; }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status) { return false; }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers)) { return false; }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    mNumber = rand() % 10;
    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(),
                inputH, inputW);

    //// Print an ascii representation
    // LINFOC("SampleOnnxMNIST", "Input:");
    // for (int i = 0; i < inputH * inputW; i++) {
    //	LINFOC("SampleOnnxMNIST", (" .:-=+*#%@"[fileData[i] / 26])
    //								  << (((i + 1) % inputW) ? "" :
    //"\n"));
    //}

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0f - float(fileData[i] / 255.0f);
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    float*    output     = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float     val{0.0f};
    int       idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    // LINFOC("SampleOnnxMNIST", "Output:");
    // for (int i = 0; i < outputSize; i++)
    //{
    //    output[i] /= sum;
    //    val = std::max(val, output[i]);
    //    if (val == output[i]) { idx = i; }

    //    LINFOC("SampleOnnxMNIST", " Prob "
    //                                  << i << "  " << std::fixed << std::setw(5)
    //                                  << std::setprecision(4) << output[i] << " "
    //                                  << "Class " << i << ": "
    //                                  << std::string(int(std::floor(output[i] * 10 + 0.5f)),
    //                                  '*'));
    //}

    return idx == mNumber && val > 0.9f;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleParams initializeSampleParams(std::string dataDir)
{
    SampleParams params;
    params.dataDirs.push_back(dataDir);

    // params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.batchSize = 1;
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = false;
    params.int8    = false;
    params.fp16    = false;

    return params;
}

} // namespace eztrt
