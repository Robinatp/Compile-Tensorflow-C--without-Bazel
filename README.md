# Compile-Tensorflow-C--without-Bazel

Compile Tensorflow C++ without Bazel, You could create a C++ Tensorflow project in your favorite C++ IDEs and build it with Makefile or CMake and you will need to do some extra work to allow gcc to be able to compile successfully C++ Tensorflow codes. So you don't have to compile  with bazel! It also approachs a method that Import OpenCV Mat into C++ Tensorflow without copying.Just convert Mat to Tensor for tensorflow run function


# This work is inspired by [tuanphuc](https://tuanphuc.github.io/standalone-tensorflow-cpp/),Thanks for his blog!


# tensorflow_cpp
  This dir include some *.h and script for merge or generate tensorflow head file for compiling


# tensorflow_classify_cpp
  This is a project of Eclipse for classifing object by inception_v3_2016_08_28_frozen.pb,most of important is that converting the Mat of opencv to tensor of Tensorflow correctly.I have introduced some functions for your calling.You can simply call my function for geting a tensor which can be used by tensorflow run function.

# tensorflow_detection_cpp
   This is a project of Eclipse for object detection by frozen_inference_graph.pb.It also suggestes a method that converting OpenCV Mat into C++ Tensorflow without copying.
