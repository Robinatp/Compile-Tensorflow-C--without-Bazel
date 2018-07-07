#ifndef TF_C_TEST_COMMONFUNCS_H
#define TF_C_TEST_COMMONFUNCS_H

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include "Detection.h"


std::map<int, std::string> ReadLabelsFromPbtxt(std::string labelFilePath,
                                               std::string item_id_name = "id",
                                               std::string item_display_name = "display_name");

std::map<int, std::string> ReadLabelsFromTxt(std::string labelFilePath);

std::string GetClassNameById(int class_id, std::map<int, std::string> labelMap);

void DrawBoxOnPic(const std::vector<Detection> dets, cv::Mat img);

cv::Rect GetRectFromRect2d(cv::Rect2d rect2d, int width, int height);


#endif //TF_C_TEST_COMMONFUNCS_H
