/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.
#ifndef CLASSIFY_H
#define CLASSIFY_H

#include <fstream>
#include <utility>
#include <vector>
#include <stdio.h>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

//#pragma mark - Includes: OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>


using namespace cv;

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

#define INPUT_WIDTH  (299)
#define INPUT_HEIGHT (299)


int classify_pb_init(std::unique_ptr<tensorflow::Session>  &session);

void convertotensor(Mat input,tensorflow::Tensor &image_tensor,Size size);

Status MatToTensorOfUint8_1(Mat input , const int input_height,
		const int input_width, const float input_mean,
		const float input_std, tensorflow::Tensor &out_tensors);


Status MatToTensorOfFloat_1(Mat input , const int input_height,
		const int input_width, const float input_mean,
		const float input_std, tensorflow::Tensor &out_tensors);

Status MatToTensorOfFloat_2(Mat input , const int input_height,
		const int input_width, const float input_mean,
		const float input_std, tensorflow::Tensor &out_tensors);

Status MatToTensorOfFloat_3(Mat input , const int input_height,
		const int input_width, const float input_mean,
		const float input_std, tensorflow::Tensor &out_tensors);

Status ReadTensorFromImageFile_by_opencv(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
							   tensorflow::Tensor* out_tensors);



Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count);


static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output);


Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors);

Status LoadGraph(const string& graph_file_name,
		std::unique_ptr<tensorflow::Session>* session);


Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores);


Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      const string& labels_file_name);


Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected);


#endif //CLASSIFY_H
