#ifndef TF_C_TEST_TFDETECTOR_H
#define TF_C_TEST_TFDETECTOR_H

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <map>

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
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/opencv.hpp>
#include "Detection.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;

struct TFModelPara {
    string graph_path;
    string labels_path;
    int32 input_width;
    int32 input_height;
    float input_mean;
    float input_std;
    string input_layer;
    vector<string> output_layer;
};

class CXXTFDetector {
public:
    CXXTFDetector(TFModelPara modelPara);
    ~CXXTFDetector();

    void InitCXXTFDetector(bool flag_use_gpu = false);
    // detect disk files
    std::vector<Detection> Detect(const string image_path, const float conf_thresh);
    // detect cv::Mat
    std::vector<Detection> Detect(cv::Mat image, const float conf_thresh);

private:
    Status LoadGraph(const string& graph_file_name,
                     std::unique_ptr<tensorflow::Session>* session,
                     bool flag_use_gpu = false);
    Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                                   const int input_width, const float input_mean,
                                   const float input_std, std::vector<Tensor>* out_tensors);
    static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                                 Tensor* output);

    // read labels
    Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                          size_t* found_label_count);

private:
    std::unique_ptr<tensorflow::Session> m_session;
    TFModelPara m_modelPara;
    std::map<int, std::string> m_labelMap;

};


#endif //TF_C_TEST_TFDETECTOR_H
