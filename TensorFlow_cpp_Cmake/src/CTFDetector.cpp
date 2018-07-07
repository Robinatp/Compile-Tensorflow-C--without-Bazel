#include "CTFDetector.h"
#include "Common.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

CTFDetector::CTFDetector()
{
    m_status = TF_NewStatus();
    m_graph = TF_NewGraph();
}

CTFDetector::~CTFDetector()
{
    TF_CloseSession(m_session, m_status);
    assert(TF_GetCode(m_status) == TF_OK);
    TF_DeleteSession(m_session, m_status);
    assert(TF_GetCode(m_status) == TF_OK);
    TF_DeleteStatus(m_status);
    TF_DeleteGraph(m_graph);
}

void CTFDetector::InitCTFDetector(const char *model_path, const char *labels_path)
{
    this->LoadGraph(model_path);
    m_labelMap = ReadLabelsFromPbtxt(labels_path);

    // setup graph inputs
    TF_Operation *placeholder = TF_GraphOperationByName(m_graph, "image_tensor");
    m_inputs.push_back({placeholder, 0});

    // setup graph outputs
    TF_Operation *output_op0 = TF_GraphOperationByName(m_graph, "detection_boxes");
    TF_Operation *output_op1 = TF_GraphOperationByName(m_graph, "detection_scores");
    TF_Operation *output_op2 = TF_GraphOperationByName(m_graph, "detection_classes");
    TF_Operation *output_op3 = TF_GraphOperationByName(m_graph, "num_detections");
    m_outputs.push_back({output_op0, 0});
    m_outputs.push_back({output_op1, 0});
    m_outputs.push_back({output_op2, 0});
    m_outputs.push_back({output_op3, 0});
}

std::vector<Detection> CTFDetector::Detect(cv::Mat image, const float conf_thresh)
{
    std::vector<Detection> detectionVec;

    // create image tensor
    const int64_t tensorDims[4] = {1, image.rows, image.cols, 3};
    TF_Tensor *input_tensor = TF_NewTensor(TF_UINT8, tensorDims, 4, image.data,
                                           size_t(image.cols * image.rows * 3), nullptr, nullptr);
    std::vector<TF_Tensor *> input_values;
    input_values.push_back(input_tensor);
    std::vector<TF_Tensor *> output_values(m_outputs.size(), nullptr);

    // session run
    TF_SessionRun(m_session, nullptr,
                  &m_inputs[0], &input_values[0], int(m_inputs.size()),
                  &m_outputs[0], &output_values[0], int(m_outputs.size()),
                  nullptr, 0, nullptr, m_status);
    if (TF_GetCode(m_status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to run session %s", TF_Message(m_status));
        exit(-1);
    }
//    TF_Tensor *res_tensor = output_values[0];
//    LOG(ERROR) << TF_NumDims(res_tensor) << " " << TF_TensorType(res_tensor);
//    for(int i = 0; i < TF_NumDims(res_tensor); ++i) {
//        LOG(ERROR) << TF_Dim(res_tensor, i);
//    }

    // collect detect results
    float *values0 = static_cast<float *>(TF_TensorData(output_values[0]));
    float *values1 = static_cast<float *>(TF_TensorData(output_values[1]));
    float *values2 = static_cast<float *>(TF_TensorData(output_values[2]));
    float *values3 = static_cast<float *>(TF_TensorData(output_values[3]));
    int numDetections = int(values3[0]);
    for (int i = 0; i < numDetections; ++i) {
        float det_score = values1[i];
        if (det_score > conf_thresh) {
            Detection tmp_dection;
            int det_class = int(values2[i]);
            std::string det_class_name = GetClassNameById(det_class, m_labelMap);
            cv::Rect2d rect2d(cv::Point2d(values0[i * 4 + 1], values0[i * 4]),
                              cv::Point2d(values0[i * 4 + 3], values0[i * 4 + 2]));
            tmp_dection.setRect2d(rect2d);
            tmp_dection.setScore(det_score);
            tmp_dection.setClass(det_class_name);
            detectionVec.push_back(tmp_dection);
        }
    }

    for (int i = 0; i < output_values.size(); ++i)
        TF_DeleteTensor(output_values[i]);
    // Todo: what's the matter with this?
//    for (int i = 0; i < input_values.size(); ++i)
//        TF_DeleteTensor(input_values[i]);

    return detectionVec;
}

void CTFDetector::LoadGraph(const char * model_path)
{
    TF_Buffer* graph_def = read_file(model_path);

    // import graph_def into graph
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(m_graph, graph_def, opts, m_status);
    TF_DeleteImportGraphDefOptions(opts);
    if (TF_GetCode(m_status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(m_status));
        exit(-1);
    }
    printf("Successfully imported graph. \n");
    TF_DeleteBuffer(graph_def);

    // create session
    TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
    m_session = TF_NewSession(m_graph, sessionOptions, m_status);
    if (TF_GetCode(m_status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to create session %s", TF_Message(m_status));
        exit(-1);
    }
    printf("Successfully created session. \n");
    TF_DeleteSessionOptions(sessionOptions);
}

TF_Buffer* CTFDetector::read_file(const char *file)
{
    FILE *f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  //same as rewind(f);

    void* data = malloc(fsize);
    size_t res = fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = this->free_buffer;
    return buf;
}

void CTFDetector::free_buffer(void *data, size_t length)
{
    free(data);
}
