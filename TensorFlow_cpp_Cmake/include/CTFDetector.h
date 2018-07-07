#ifndef TF_C_TEST_CTFDETECTOR_H
#define TF_C_TEST_CTFDETECTOR_H

//The CTFDetector only needs c_api.h
#include "tensorflow/c/c_api.h"

#include <stdio.h>
#include <stdlib.h>
//#include <assert.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Detection.h"


class CTFDetector {
public:
    CTFDetector();
    ~CTFDetector();

    void InitCTFDetector(const char * model_path, const char *labels_path);

    std::vector<Detection> Detect(cv::Mat image, const float conf_thresh);

private:
    void LoadGraph(const char * model_path);

private:
    TF_Buffer* read_file(const char* file);

    static void free_buffer(void* data, size_t length);

private:
    TF_Status* m_status;
    TF_Graph* m_graph;
    TF_Session* m_session;

    std::vector<TF_Output> m_inputs, m_outputs;
    std::map<int, std::string> m_labelMap;

};


#endif //TF_C_TEST_CTFDETECTOR_H
