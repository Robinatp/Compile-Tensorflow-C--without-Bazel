#include <sys/time.h>
#include "CTFDetector.h"
#include "Common.h"


int main()
{
    printf("Hello from TensorFlow C library version %s\n", TF_Version());

    CTFDetector my_detector;
    my_detector.InitCTFDetector("../models/frozen_inference_graph.pb",
                                "../models/labels/mscoco_label_map.pbtxt");

    timeval start_time, end_time;
    while(1)
    {
        cv::Mat img = cv::imread("../data/001257.jpg");
        gettimeofday(&start_time, NULL);
        std::vector<Detection> detectionVec = my_detector.Detect(img, 0.5);
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






//// Put an image in the cameraImg mat
//cv::resize(image->image, cameraImg, cv::Size(inputwidth, inputheight), 0, 0, cv::INTER_AREA);
//// Create a new tensor pointing to that memory:
//const int64_t tensorDims[4] = {1,inputheight,inputwidth,3};
//int *imNumPt = new int(1);
//TF_Tensor* tftensor = TF_NewTensor(TF_DataType::TF_UINT8, tensorDims, 4,
//                                   cameraImg.data, inputheight * inputwidth * 3,
//                                   NULL, imNumPt);
//Tensor inputImg = tensorflow::TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);

//int main() {
//
//    TF_Status *s = TF_NewStatus();
//    TF_Graph *graph = TF_NewGraph();
//
//    const char *graph_def_data; // <-- your serialized GraphDef here
//    TF_Buffer graph_def = {graph_def_data, strlen(graph_def_data), nullptr};
//
//// Import `graph_def` into `graph`
//    TF_ImportGraphDefOptions *import_opts = TF_NewImportGraphDefOptions();
//    TF_ImportGraphDefOptionsSetPrefix(import_opts, "import");
//    TF_GraphImportGraphDef(graph, &graph_def, import_opts, s);
//    assert(TF_GetCode(s) == TF_OK);
//
//// Setup graph inputs
//    std::vector<TF_Output> inputs;
//    std::vector<TF_Tensor *> input_values;
//// Add the placeholders you would like to feed, e.g.:
//    TF_Operation *placeholder = TF_GraphOperationByName(graph, "import/my_placeholder");
//    inputs.push_back({placeholder, 0});
//    TF_Tensor *tensor = TF_NewTensor(/*...*/);
//    input_values.push_back(tensor);
//
//// Setup graph outputs
//    std::vector<TF_Output> outputs;
//// Add the node outputs you would like to fetch, e.g.:
//    TF_Operation *output_op = TF_GraphOperationByName(graph, "import/my_output");
//    outputs.push_back({output_op, 0});
//    std::vector<TF_Tensor *> output_values(outputs.size(), nullptr);
//
//// Run `graph`
//    TF_SessionOptions *sess_opts = TF_NewSessionOptions();
//    TF_Session *session = TF_NewSession(graph, sess_opts, s);
//    assert(TF_GetCode(s) == TF_OK);
//    TF_SessionRun(session, nullptr,
//                  &inputs[0], &input_values[0], inputs.size(),
//                  &outputs[0], &output_values[0], outputs.size(),
//                  nullptr, 0, nullptr, s);
//
//    void *output_data = TF_TensorData(output_values[0]);
//    assert(TF_GetCode(s) == TF_OK);
//
//// If you have a more complicated workflow, I suggest making scoped wrapper
//// classes that call these in their destructors
//    for (int i = 0; i < inputs.size(); ++i) TF_DeleteTensor(input_values[i]);
//    for (int i = 0; i < outputs.size(); ++i) TF_DeleteTensor(output_values[i]);
//    TF_CloseSession(session, s);
//    TF_DeleteSession(session, s);
//    TF_DeleteSessionOptions(sess_opts);
//    TF_DeleteImportGraphDefOptions(import_opts);
//    TF_DeleteGraph(graph);
//    TF_DeleteStatus(s);
//}
