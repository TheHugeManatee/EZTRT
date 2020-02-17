#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace eztrt
{
class model;

/**
 * Computes the softmax function for the specified input along the specified dimension.
 */
cv::Mat softmax(cv::Mat in, int dim = 1);

/**
 * Applies a set of preprocessing operations that can be flexibly defined through the content of the
 * `step_list` parameter.
 *
 */
cv::Mat apply_preprocess_steps(cv::Mat in, std::string step_list);

/**
 * Shows all channels of a channel-separated result batch (i.e. a result of shape $[1,C,H,W]$).
 *
 * Uses cv::imshow for all outputs. Does not perform any remapping of the values to a reasonable
 * output range, so make sure that happens beforehand.
 */
void show_all_channels(cv::Mat result);

/**
 * Saves all channels of a channel-separated result batch $[1,C,H,W]$ to a file.
 *
 * Filename should continue a brace placeholder in case `result` has more than one channel $C$,
 * i.e. "result_C{}.png". Saves each channel as a grayscale image. Does not do any rescaling, thus
 * make sure you remap the value range to an appropriate range in case you save to a format that
 * expects [0,255] pixel ranges.
 */
void save_all_channels(cv::Mat result, std::string file_base);

/**
 * Reshapes a Mat to make the channels an explicit dimension. I.e. a
 * 3-channel image of (128,64) size will result in a (128,64,3) 1-channel image.
 */
cv::Mat reshape_channels(cv::Mat m);

/**
 * Reorders the dimensions in-place and returns a new cv::Mat header with the appropriate shape.
 * This is equivalent to for example pytorch's tensor.permute, and actually copies data **in
 * place**. The returned `cv::Mat` is merely a new matrix header to the same data as `m`, but
 * permuted. For example, to permute an input `m` from $H,W,C$ order to channel separated order
 * $[C,W,H]$, use `auto permuted = permute_dims(m, {2,0,1});`.
 *
 * \param m			The input matrix to permute
 * \param new_order The new order of dimensions
 * \return			A new matrix header to the permuted data of `m`
 */
cv::Mat permute_dims(cv::Mat m, std::vector<int> new_order);

/**
 * Attempts to automatically adjust the shape of the given input to what the model expects.
 * Currently only works with a 2D image as an input and a model that expects a [N,H,W,C] input which
 *is the usual input shape for computer vision networks.
 *
 * Tries to guess what to do based on some heuristics:
 *  - tries to adapt the element type according to the expected input:
 *	   FLOAT: Values are REMAPPED to [-1,1] for a signed input and [0,1] for an unsigned input
 *     INT8: Values are REMAPPED from integer formats. Floating point remaps [-1,1] to [-127,127]
 *     INT32: Values are CONVERTED from integer and floating point formats
 *  - If H,W do not match the rows and cols of the input, use `cv::resize` to resample the input
 *  - if image channels do not match C, use `cv::cvtColor` where possible to adapt the number of
 *channels
 *  - makes the channel an explicit dimension and reshapes the channels to the normal input, i.e.
 *transforms from the standard OpenCV layout of [H,W,C] (interleaved channels) to [C,H,W] (separated
 *channels)
 *
 * \return a `cv::Mat` that should have a shape that can be passed directly to `m.predict()`. If
 *this method was not successful, will return an emtpy matrix.
 */
cv::Mat try_adjust_input(cv::Mat input, int input_index, model& m);

} // namespace eztrt
