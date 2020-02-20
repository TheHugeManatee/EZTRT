# EZTRT - Wrappers for NVidia TensorRT for easy use with OpenCV
Wrapper library for NVidia TensorRT library for easy use with OpenCV

## Setup
Compile and install the project using CMake.

You will need:
 - [TensorRT](https://developer.nvidia.com/tensorrt) and CuDNN
 - CUDA (Probably anthing as recent as 9.0 should work)

Currently tested on Windows (MSVC 2017), TensorRT 7.0, CUDA 10.2. Other platforms may work.

## Usage
After `make` and `make install`, import in your cmake-based project with
```CMake
    find_package(eztrt REQUIRED)
    target_link_libraries(your-project eztrt::eztrt)
```

Within your code, use the library as:
```C++
// Create a logger object - you can also subclass logger and override log()
eztrt::logger log("Main", ILogger::Severity::kVERBOSE);

// some configurable metaparameters for building the model
model::params params;

model m(params, log);

// load an ONNX model, for example from https://github.com/onnx/models/tree/master/vision/classification/resnet
m.load("resnetv1.onnx");

// load an example image
cv::Mat input = cv::imread("example.png");

// automatically reorder and resize the input so it fits into the expected input tensor of the model
cv::Mat in_data = try_adjust_input(input, 0, m);

// predict an output
cv::Mat out_data = m.predict(in_data);

// if you have a classification/segmentation problem, you might want to softmax your output:
cv::Mat softmaxed = eztrt::softmax(out_data);

// for a resnet, softmaxed now is a [1,1000] float matrix with the class probabilities for ImageNet classes
```

## TRT Host
A simple example application that hosts a model and performs inference on a video/image stream is included.

## Other things
There are some additional helper functions to show multi-channel tensor outputs and convert back and forth between OpenCV and Tensor layout.

To reduce loading times, you can serialize an optimized model ("engine" in TensorRT terms) to speed up loading the next time:
```C++
model m(params, log);
m.load("resnetv1.onnx");
m.serialize_engine("resnetv1.blob");

// to re-load it:
m.load("resnetv1.onnx", "resnetv1.blob");
```

## TODO/Limitations
Currently the most basic functionality works: One single input, one single output. Models that expect several outputs are not direclty supported (multiple outputs can be read by accessing `model::outputs` though)

 - Loading a serialized engine should not require the model definition (ONNX) file anymore. 
 - Extend to multiple inputs and outputs
 - support different element data types (currently everything is converted to float, not even sure what happens with a model that expects different a different `dtype`)
 - batch processing (currently only single-shot inference is possible)
 - asynchronous processing
 - lots more convenience and stability things
 - Support `opencv::cuda` cv::Mats
 - More interoperability (OpenGL interop to read/write textures directly)