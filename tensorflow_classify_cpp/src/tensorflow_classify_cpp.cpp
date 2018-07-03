//============================================================================
// Name        : tensorflow_classify_cpp.cpp
// Author      : robin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "classify.h"
//using namespace std;


int main( int argc, char** argv )
{
  Mat cv_image;
  namedWindow( "Display Image", WINDOW_AUTOSIZE );
  //cv_image = imread("/home/ubuntu/eclipse-workspace-cpp/tensorflow_demo/data/construction_workers_images/001300.jpg");
  cv_image = imread("/home/ubuntu/eclipse-workspace-cpp/tensorflow-cpp/data/grace_hopper.jpg");



  std::unique_ptr<tensorflow::Session>  session;

  if( classify_pb_init(session) <0){
	  LOG(ERROR) << "graph init faile";
	  return -1;
  }

  string labels = "/home/ubuntu/eclipse-workspace/facenet/tensorflow_cpp/data/imagenet_slim_labels.txt";
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";


  //
  tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
  ReadTensorFromImageFile_by_opencv("/home/ubuntu/eclipse-workspace-cpp/tensorflow-cpp/data/grace_hopper.jpg",
		  INPUT_HEIGHT,INPUT_WIDTH,
		  0, 255,
		  &image_tensor);


  //method for convert Mat to Tensor
//  tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
//  MatToTensorOfFloat_3(cv_image , INPUT_HEIGHT, INPUT_WIDTH, 0, 255, image_tensor);


    //method for convert Mat to Tensor
//  tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
//  MatToTensorOfFloat_2(cv_image , INPUT_HEIGHT, INPUT_WIDTH, 0, 255, image_tensor);




  //method for convert Mat to Tensor
//  tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
//  MatToTensorOfFloat_1(cv_image , INPUT_HEIGHT, INPUT_WIDTH, 0, 255, image_tensor);


  //method for convert Mat to Tensor
//  tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
//  convertotensor(cv_image, image_tensor, Size(INPUT_HEIGHT, INPUT_WIDTH));



  imshow( "Display Image", cv_image );
  waitKey(0);

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, image_tensor}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }



  // Do something interesting with the results we've generated.
  Status print_status = PrintTopLabels(outputs, labels);

  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }

  return 0;
}

