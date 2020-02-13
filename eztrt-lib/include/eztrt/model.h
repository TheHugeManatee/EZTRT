#pragma once

#include "eztrt/base.h"
#include "eztrt/buffers.h"
#include "eztrt/common.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <opencv2/opencv.hpp>

namespace eztrt
{
class logger;

class model
{
public:
    struct params
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

    model(params params, logger& logger);

    // Non-copyable
    model(const model& rhs) = delete;
    model& operator=(const model& rhs) = delete;
    // Movable
    model(model&& rhs) = default;
    model& operator=(model&& rhs) = default;

    ~model() = default;

    bool predict(cv::Mat input);

    std::string summarize();

    /**
     * Creates the engine object. Call this only after network and config are set.
     */
    bool create_engine();

    /**
     * Check if the model is valid for inference. If it is not, check `summarize()` for reasons.
     */
    bool valid();

    /**
     * Parse a file (currently only ONNX is supported)
     */
    bool load(std::string file);

    void apply_params();

    nvinfer1::INetworkDefinition&          network() { return *network_; }
    nvinfer1::IBuilder&                    builder() { return *builder_; }
    nvinfer1::IBuilderConfig&              config() { return *config_; }
    std::shared_ptr<nvinfer1::ICudaEngine> engine() { return engine_; };
    void set_engine(std::shared_ptr<nvinfer1::ICudaEngine> engine) { engine_ = engine; }

private:
    eztrt::InferUniquePtr<nvinfer1::IBuilder>           builder_;
    eztrt::InferUniquePtr<nvinfer1::INetworkDefinition> network_;
    eztrt::InferUniquePtr<nvinfer1::IBuilderConfig>     config_;
    std::shared_ptr<nvinfer1::ICudaEngine>              engine_;
    InferUniquePtr<nvinfer1::IExecutionContext>         context_;

    logger& logger_;

    params params_;
};

} // namespace eztrt
