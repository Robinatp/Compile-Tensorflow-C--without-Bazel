#include"object_detection.h"


// Helper Functions: High Level

Status MatToTensorOfUint8_1(Mat input , const int input_height,
		const int input_width, const float input_mean,
		const float input_std, tensorflow::Tensor &out_tensors) {
	// resize
	resize(input, input, Size(input_height, input_width));
	// color convert
	cvtColor(input, input, COLOR_BGR2RGB);

    //tensorflow::Tensor imgTensorWithSharedData(tensorflow::DT_UINT8, {1, input_height, input_width, input->channels()});
    uint8_t *p = out_tensors.flat<uint8_t>().data();
    Mat outputImg(input_width, input_mean, CV_8UC3, p);
    input.convertTo(outputImg, CV_8UC3);

	return Status::OK();
}

// Draws bounding boxes on the cvImg and returns the number of predictions
// helpful code at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc
int drawPredictions(Mat * cvImg, std::unique_ptr<tensorflow::Session>* tf_session, const float predictionThreshold, std::map<int, std::string> TF_LabelMap) {
    // avoid memory copies using strategy from https://github.com/tensorflow/tensorflow/issues/8033#issuecomment-332029092
    int inputHeight = cvImg->size().height;
    int inputWidth = cvImg->size().width;


    tensorflow::Tensor imgTensorWithSharedData(tensorflow::DT_UINT8, {1, inputHeight, inputWidth, cvImg->channels()});
    uint8_t *p = imgTensorWithSharedData.flat<uint8_t>().data();
    Mat outputImg(inputHeight, inputWidth, CV_8UC3, p);
    // color convert

    cvImg->convertTo(outputImg, CV_8UC3);
    cvtColor(*cvImg, outputImg, COLOR_BGR2RGB);

    // Run tensorflow
    TickMeter tm;
    tm.start();
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = (*tf_session)->Run({{"image_tensor:0", imgTensorWithSharedData}},
                                                       {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"},
                                                       {},
                                                       &outputs);
    if (!run_status.ok()) {
        std::cerr << "TF_Session->Run Error: " << run_status << std::endl;
    }
    tm.stop();
    std::cout << "Inference time, ms: " << tm.getTimeMilli()  << std::endl;

    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float,3>();

    int detectionsCount = (int)(num_detections(0));
    int drawnDetections = 0;
    RNG rng(12345);
    std::cout << "Total detections before threshold: " << detectionsCount << std::endl;
    for(int i = 0; i < detectionsCount && i < 100000; ++i) { // 100000 is infinite loop protection
        if(scores(i) > predictionThreshold) {
            float boxClass = classes(i);

            float x1 = float(outputImg.size().width) * boxes(0,i,1);
            float y1 = float(outputImg.size().height) * boxes(0,i,0);

            float x2 = float(outputImg.size().width) * boxes(0,i,3);
            float y2 = float(outputImg.size().height) * boxes(0,i,2);

            std::ostringstream label;
            label << TF_LabelMap[boxClass] << ", confidence: " << (scores(i)  * 100) << "%";
            std::cout << "Detection " << (i+1) << ": class: " << boxClass << " " << label.str() << ", box: (" << x1 << "," << y1 << "), (" << x2 << "," << y2 << ")" << std::endl;

            Scalar randomColor = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
            rectangle(*cvImg, Point(x1, y1), Point(x2, y2), randomColor);
            putText(*cvImg, label.str(), Point(x1, y1), 1, 1.0, randomColor);
            drawnDetections++;
        }
    }

    return drawnDetections;
}

// Helper Functions: Tensorflow

// Loads the graph into the session and starts up the labels map
bool TF_init(const std::string& labels_file_name, std::map<int, std::string>* label_map, const std::string& graph_file_name, std::unique_ptr<tensorflow::Session>* tf_session) {
    int argc = 0;
    tensorflow::port::InitMain(NULL, &argc, NULL);

    int label_count;
    if (!TF_loadLabels(labels_file_name, label_map, &label_count)) {
        std::cerr << "TF_loadLabels ERROR" << std::endl;
    } else {
        std::cout << "Loaded " << label_count << " dnn class labels" << std::endl;
    }

    tensorflow::Status status = TF_LoadGraph(graph_file_name, tf_session);
    if (!status.ok()) {
        std::cerr << "TF_LoadGraph ERROR: " << status.error_message() << std::endl;
        return false;
    }
    return true;
}

// Takes a file name, and loads a list of labels from it, one per line into the map object. Expects `CLASSID: CLASSNAME` fmt
bool TF_loadLabels(const std::string& file_name, std::map<int, std::string>* result, int* found_label_count) {
    std::ifstream file(file_name);
    if (!file) {
        return false;
    }
    result->clear();
    *found_label_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::string::size_type sz;   // alias of size_t
        int i_decimal = std::stoi(line, &sz);
        (*result)[i_decimal] = line.substr(sz+2); // +2 to account for ':' and following space
        (*found_label_count)++;
    }
    return true;
}

// Reads a model graph definition from disk, and creates a session object you can use to run it.
tensorflow::Status TF_LoadGraph(const std::string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status =
    tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    return (*session)->Create(graph_def);
}
