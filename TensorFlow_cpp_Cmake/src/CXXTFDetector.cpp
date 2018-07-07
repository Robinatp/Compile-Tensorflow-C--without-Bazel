#include "CXXTFDetector.h"
#include "Common.h"
//#include <stdint.h>

//#define Debug ;

CXXTFDetector::CXXTFDetector(TFModelPara modelPara)
{
    m_modelPara = modelPara;
}

CXXTFDetector::~CXXTFDetector()
{
}

void CXXTFDetector::InitCXXTFDetector(bool flag_use_gpu)
{
    // load and initialize the model.
    LOG(INFO) << "Loading graph: " << m_modelPara.graph_path;
    Status load_graph_status = LoadGraph(m_modelPara.graph_path, &m_session, flag_use_gpu);
    if (!load_graph_status.ok()) {
        LOG(FATAL) << "LoadGraph ERROR!!!!"<< load_graph_status;
    }

    m_labelMap = ReadLabelsFromPbtxt(m_modelPara.labels_path);
//    m_labelMap = this->ReadLabelsFromTxt(modelPara.labels_path);
    LOG(INFO) << "Read " << m_labelMap.size() << " labels from " << m_modelPara.labels_path;

#ifdef Debug
    std::vector<tensorflow::DeviceAttributes> response;
    m_session->ListDevices(&response);
    for(int i = 0; i < response.size(); ++i){
        LOG(ERROR) << response[i].DebugString();
    }
#endif
}

std::vector<Detection> CXXTFDetector::Detect(const string image_path, const float conf_thresh)
{
    std::vector<Detection> detectionVec;

    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main graph expects.
    std::vector<Tensor> resized_tensors;
    Status read_tensor_status =
            this->ReadTensorFromImageFile(image_path, m_modelPara.input_height,
                                          m_modelPara.input_width, m_modelPara.input_mean,
                                          m_modelPara.input_std, &resized_tensors);
    if (!read_tensor_status.ok()) {
        LOG(FATAL) << read_tensor_status;
    }
    const Tensor& resized_tensor = resized_tensors[0];

#ifdef Debug
    LOG(ERROR) <<"image shape:" << resized_tensor.shape().DebugString()
               << ",len:" << resized_tensors.size()
               << ",tensor type:"<< resized_tensor.dtype();
    // << ",data:" << resized_tensor.flat<tensorflow::uint8>();
#endif

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = m_session->Run({{m_modelPara.input_layer, resized_tensor}},
                                       m_modelPara.output_layer, {}, &outputs);
    if (!run_status.ok()) {
        LOG(FATAL) << "Running model failed: " << run_status;
    }

#ifdef Debug
    int image_width = resized_tensor.dims();
    int image_height = 0;
    //int image_height = resized_tensor.shape()[1];

    LOG(ERROR) << "size:" << outputs.size() << ",image_width:" << image_width
               << ",image_height:" << image_height << endl;
#endif

    //tensorflow::TTypes<float>::Flat iNum = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float,3>();

#ifdef Debug
    LOG(ERROR) << "num_detections:" << num_detections(0) << ","
               << outputs[0].shape().DebugString();
#endif

    for(size_t i = 0; i < num_detections(0); ++i) {
        if(scores(i) > conf_thresh) {
            Detection tmp_dection;
            cv::Rect2d rect2d(cv::Point2d(boxes(0,i,1), boxes(0,i,0)),
                              cv::Point2d(boxes(0,i,3), boxes(0,i,2)));
            string class_name = GetClassNameById(classes(i), m_labelMap);
            tmp_dection.setClass(class_name);
            tmp_dection.setRect2d(rect2d);
            tmp_dection.setScore(scores(i));
            detectionVec.push_back(tmp_dection);
//            LOG(ERROR) << "class:" << class_name << ", score:" << scores(i);
        }
    }

    return detectionVec;
}

std::vector<Detection> CXXTFDetector::Detect(cv::Mat image, const float conf_thresh)
{
    std::vector<Detection> detectionVec;
    int width = image.cols;
    int height = image.rows;
    int depth = image.channels();

//    cv::Mat draw_image = image.clone();
//    cv::resize(image, image, cv::Size(m_modelPara.input_width, m_modelPara.input_height));
    // creating a Tensor for storing the data
    tensorflow::Tensor input_tensor(tensorflow::DT_UINT8,
                                    tensorflow::TensorShape({1, height, width, depth}));
    auto input_tensor_mapped = input_tensor.tensor<uint8_t, 4>();

//    cv::Mat Image2;
//    image.convertTo(Image2, CV_32FC1);
//    image = Image2;
    image = image - m_modelPara.input_mean;
    image = image / m_modelPara.input_std;
    const uint8_t * source_data = (uint8_t*) image.data;

    // copying the data into the corresponding tensor
//    for (int y = 0; y < m_modelPara.input_height; ++y) {
//        const uint8_t* source_row = source_data + (y * m_modelPara.input_width * depth);
//        for (int x = 0; x < m_modelPara.input_width; ++x) {
//            const uint8_t* source_pixel = source_row + (x * depth);
//            for (int c = 0; c < depth; ++c) {
//                const uint8_t* source_value = source_pixel + c;
//                input_tensor_mapped(0, y, x, c) = *source_value;
//            }
//        }
//    }
    for (int y = 0; y < height; ++y) {
        const uint8_t* source_row = source_data + (y * width * depth);
        for (int x = 0; x < width; ++x) {
            const uint8_t* source_pixel = source_row + (x * depth);
            const uint8_t* source_B = source_pixel + 0;
            const uint8_t* source_G = source_pixel + 1;
            const uint8_t* source_R = source_pixel + 2;

            input_tensor_mapped(0, y, x, 0) = *source_R;
            input_tensor_mapped(0, y, x, 1) = *source_G;
            input_tensor_mapped(0, y, x, 2) = *source_B;
        }
    }

    std::vector<Tensor> outputs;
    Status run_status = m_session->Run({{m_modelPara.input_layer, input_tensor}},
                                       m_modelPara.output_layer, {}, &outputs);
    if (!run_status.ok()) {
        LOG(FATAL) << "Running model failed: " << run_status;
    }

    //tensorflow::TTypes<float>::Flat iNum = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float,3>();


    for(size_t i = 0; i < num_detections(0); ++i) {
        if(scores(i) > conf_thresh) {
            Detection tmp_dection;
            cv::Rect2d rect2d(cv::Point2d(boxes(0,i,1), boxes(0,i,0)),
                              cv::Point2d(boxes(0,i,3), boxes(0,i,2)));
            string class_name = GetClassNameById(classes(i), m_labelMap);
            tmp_dection.setClass(class_name);
            tmp_dection.setRect2d(rect2d);
            tmp_dection.setScore(scores(i));
            detectionVec.push_back(tmp_dection);
//            LOG(ERROR) << "class:" << class_name << ", score:" << scores(i);
        }
    }

    return detectionVec;
}

// read graph and create session
Status CXXTFDetector::LoadGraph(const string &graph_file_name,
                             std::unique_ptr<tensorflow::Session> *session,
                             bool flag_use_gpu)
{
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }

    // TODO: set cpu or gpu here, now it doesn't work
//    tensorflow::SessionOptions session_opts;
//    for(int i = 0; i < graph_def.node_size(); ++i) {
//        auto node = graph_def.mutable_node(i);
//        node->set_device("/cpu:0");
//        LOG(ERROR) << node->device() << endl;
//    }
//    if(flag_use_gpu) {
//        tensorflow::graph::SetDefaultDevice("/device:GPU:0", &graph_def);
//        session_opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
//        session_opts.config.mutable_gpu_options()->set_allow_growth(true);
//    } else {
//        tensorflow::graph::SetDefaultDevice("/cpu:0", &graph_def);
//        session_opts.config.add_device_filters("/cpu:0");
//    }
//    session->reset(tensorflow::NewSession(session_opts));

    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status CXXTFDetector::ReadTensorFromImageFile(const string &file_name, const int input_height,
                                           const int input_width, const float input_mean,
                                           const float input_std, std::vector<Tensor> *out_tensors)
{
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
            ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader =
            Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::str_util::EndsWith(file_name, ".png")) {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                 DecodePng::Channels(wanted_channels));
    } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
        // gif decoder returns 4-D tensor, remove the first dim
        image_reader =
                Squeeze(root.WithOpName("squeeze_first_dim"),
                        DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else {
        // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    // auto float_caster =
    //     Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

    auto uint8_caster =  Cast(root.WithOpName("uint8_caster"),
                              image_reader, tensorflow::DT_UINT8);

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

    // Bilinearly resize the image to fit the required dimensions.
    // auto resized = ResizeBilinear(
    //     root, dims_expander,
    //     Const(root.WithOpName("size"), {input_height, input_width}));


    // Subtract the mean and divide by the scale.
    // auto div =  Div(root.WithOpName(output_name), Sub(root, dims_expander, {input_mean}),
    //     {input_std});


    //cast to int
    //auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), div, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
            tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
    return Status::OK();
}

Status CXXTFDetector::ReadEntireFile(tensorflow::Env *env, const string &filename, Tensor *output)
{
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
    }
    output->scalar<string>()() = data.ToString();
    return Status::OK();
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status CXXTFDetector::ReadLabelsFile(const string &file_name, std::vector<string> *result,
                                  size_t *found_label_count)
{
    std::ifstream file(file_name);
    if (!file) {
        return tensorflow::errors::NotFound("Labels file ", file_name,
                                            " not found.");
    }
    result->clear();
    string line;
    while (std::getline(file, line)) {
        result->push_back(line);
    }
    *found_label_count = result->size();
    const int padding = 16;
    while (result->size() % padding) {
        result->emplace_back();
    }
    return Status::OK();
}
