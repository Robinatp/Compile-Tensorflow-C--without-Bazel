//============================================================================
// Name        : tensorflow_detection_cpp.cpp
// Author      : robin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "object_detection.h"

//# Constants: Image
String DEFAULT_IMAGE_PATH = "/home/ubuntu/eclipse-workspace-cpp/tensorflow_demo/data/construction_workers_images/001257.jpg";
String OUTPUT_IMAGE_PATH = "output.jpg";

// Constants: Tensorflow
std::string TF_PB_PATH = "/home/ubuntu/eclipse-workspace-cpp/tensorflow_demo/data/frozen_inference_graph.pb";
std::string TF_LABELLIST_PATH = "/home/ubuntu/eclipse-workspace-cpp/tensorflow_demo/data/labels.txt";
float const TF_PREDICTION_THRESSHOLD = 0.5;

// Globals: Tensorflow
std::map<int, std::string> TF_LabelMap;
std::unique_ptr<tensorflow::Session> TF_Session;


int main(int argc, const char * argv[]) {
    String imagePath = DEFAULT_IMAGE_PATH;
    if (argc >= 2) {
        imagePath = argv[1];
    }

    if (!TF_init(TF_LABELLIST_PATH, &TF_LabelMap, TF_PB_PATH, &TF_Session)) {
        return EXIT_FAILURE;
    }

    Mat image;
    image = imread(imagePath); // Read the file

    if(!image.data)  {
        std::cerr << "Could not open or find the image at " << imagePath << std::endl;
        return EXIT_FAILURE;
    }

    int detectionsCount = drawPredictions(&image, &TF_Session, TF_PREDICTION_THRESSHOLD,TF_LabelMap);
    std::cout << "Total detections: " << detectionsCount << std::endl;

    std::vector<int> compressionParams;
    compressionParams.push_back(100);
    compressionParams.push_back(90);
    imwrite(OUTPUT_IMAGE_PATH, image, compressionParams);

    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);
    waitKey(0);

    return EXIT_SUCCESS;
}
