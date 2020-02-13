#pragma once

#include "buffers.h"

#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace eztrt
{
struct InferDeleter
{
    template<typename T>
    void operator()(T* obj) const
    {
        if (obj) { obj->destroy(); }
    }
};
template<typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

class logger : public nvinfer1::ILogger
{
public:
    logger(std::string cat, Severity level = ILogger::Severity::kINFO);
    void log(Severity severity, const char* msg) override;

private:
    std::string cat_;
    Severity    level_;
};

struct SampleParams
{
    int                      batchSize{1}; //!< Number of inputs in a batch
    int                      dlaCore{-1};  //!< Specify the DLA core to run network on.
    bool                     int8{false};  //!< Allow runnning the network in Int8 mode.
    bool                     fp16{false};  //!< Allow running the network in FP16 mode.
    uint64_t                 workspace_size{0};
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{

public:
    SampleOnnxMNIST(SampleParams params, nvinfer1::ILogger& logger)
        : mParams(params), mEngine(nullptr), mLogger(logger)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(uint8_t* data);

private:
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int            mNumber{0};  //!< The number to classify
    SampleParams   mParams;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(InferUniquePtr<nvinfer1::IBuilder>&           builder,
                          InferUniquePtr<nvinfer1::INetworkDefinition>& network,
                          InferUniquePtr<nvinfer1::IBuilderConfig>&     config,
                          InferUniquePtr<nvonnxparser::IParser>&        parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    nvinfer1::ILogger& mLogger;
};

} // namespace eztrt
