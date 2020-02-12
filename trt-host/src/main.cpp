/** @file main.cpp
 * Just a simple hello world using libfmt
 */
// The previous block is needed in every file for which you want to generate documentation

#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "eztrt/bar.h"

#include <string>

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

int main(int argc, char* argv[])
{
    eztrt::Logger mLogger("Main");

    auto builder = eztrt::InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(mLogger));
    if (!builder) { return false; }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = eztrt::InferUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) { return false; }

    auto config = eztrt::InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) { return false; }

    auto parser =
        eztrt::InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
    if (!parser) { return false; }

    if (argc > 1)
    {
        if (!parser->parseFromFile(argv[1], 1))
        {
            spdlog::error("Could not successfully parse {}", argv[1]);
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                auto err = parser->getError(i);
                spdlog::error("[{}] {}", err->code(), err->desc());
            }
            return 1;
        }

        auto NumInputs = network->getNbInputs();
        for (int i = 0; i < NumInputs; ++i)
        {
            auto               input = network->getInput(i);
            std::ostringstream dims;
            dims << input->getDimensions();
            spdlog::info("Input #{}: [{}] {} {}", i, input->getName(), dims.str(),
                         to_str(input->getType()));
        }

        auto NumOutputs = network->getNbOutputs();
        for (int i = 0; i < NumOutputs; ++i)
        {
            auto               output = network->getOutput(i);
            std::ostringstream dims;
            dims << output->getDimensions();
            spdlog::info("Output #{}: [{}] {} {}", i, output->getName(), dims.str(),
                         to_str(output->getType()));
        }
    }
    return 0;
}
