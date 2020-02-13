#pragma once

#include "buffers.h"

#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "eztrt/base.h"

namespace czmtrt
{

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

} // namespace czmtrt
