#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

#include <iostream>

// Includes: OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;


// Includes: Tensorflow
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"


using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


// Declarations: High Level
int drawPredictions(Mat * cvImg, std::unique_ptr<tensorflow::Session>* tf_session, const float predictionThreshold, std::map<int, std::string> TF_LabelMap);

// Declarations: Tensorflow
bool TF_loadLabels(const std::string& file_name, std::map<int, std::string>* result, int* found_label_count);
tensorflow::Status TF_LoadGraph(const std::string& graph_file_name, std::unique_ptr<tensorflow::Session>* session);
bool TF_init(const std::string& labels_file_name, std::map<int, std::string>* label_map, const std::string& graph_file_name, std::unique_ptr<tensorflow::Session>* session);

Status MatToTensorOfUint8_1(Mat input , const int input_height,
		const int input_width, const float input_mean,
		const float input_std, tensorflow::Tensor &out_tensors);


#endif //OBJECT_DETECTION_H
