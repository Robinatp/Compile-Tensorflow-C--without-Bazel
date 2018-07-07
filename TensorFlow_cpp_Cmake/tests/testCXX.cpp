#include "CXXTFDetector.h"
#include <sys/time.h>
#include "Common.h"

int main(int argc, char* argv[])
{
    TFModelPara model_para;

    // faster_rcnn
//    model_para.graph_path ="../models/faster_rcnn_resnet101_coco/frozen_inference_graph.pb";
//    model_para.graph_path ="../models/faster_rcnn_nas_coco/frozen_inference_graph.pb";
//    model_para.input_width = 299;
//    model_para.input_height = 299;
//    const float conf_thresh = 0.7;

    // mobilenet
    model_para.graph_path = "../models/frozen_inference_graph.pb";
    model_para.input_width = 224;
    model_para.input_height = 224;
    const float conf_thresh = 0.5;

    model_para.labels_path ="../models/labels/mscoco_label_map.pbtxt";
    model_para.input_mean = 0;
    model_para.input_std = 1;
    model_para.input_layer = "image_tensor:0";
    model_para.output_layer = { "detection_boxes:0", "detection_scores:0",
                                "detection_classes:0", "num_detections:0" };

    // init TF detector
    CXXTFDetector my_detector(model_para);
    my_detector.InitCXXTFDetector();

    // TODO: *.jpg files are not supported
    string image_path = "../data/001257.jpg";

    timeval start_time, end_time;
    while(1) {
        cv::Mat img = cv::imread(image_path);
        gettimeofday(&start_time, NULL);
//        std::vector<Detection> detectionVec = my_detector.Detect(image_path, conf_thresh);
        std::vector<Detection> detectionVec = my_detector.Detect(img, conf_thresh);
        gettimeofday(&end_time, NULL);
        double timeuse = (1000000 * (end_time.tv_sec - start_time.tv_sec)
                          + end_time.tv_usec - start_time.tv_usec) / 1000000.0;
        LOG(INFO) << "Use time: " << timeuse;

        DrawBoxOnPic(detectionVec, img);

        cv::imshow("result", img);
        cv::waitKey(0);
    }

    return 0;
}
