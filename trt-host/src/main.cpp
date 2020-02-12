/** @file main.cpp
 * Just a simple hello world using libfmt
 */
// The previous block is needed in every file for which you want to generate documentation

#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "eztrt/bar.h"

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
}
