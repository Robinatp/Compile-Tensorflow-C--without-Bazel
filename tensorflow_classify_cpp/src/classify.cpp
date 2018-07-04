#include "classify.h"



int classify_pb_init(std::unique_ptr<tensorflow::Session>  &session,string graph)
{


  string root_dir = "";

  // First we load and initialize the model.
  //std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  return 0;
}


void convertotensor(Mat input,tensorflow::Tensor &image_tensor,Size size){
	resize(input, input, size);
	cvtColor(input, input, COLOR_BGR2RGB);
	//imshow( "Display Image", cv_image );
	//waitKey(0);

	//tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
	auto input_tensor_mapped = image_tensor.tensor<float, 4>();

	const uchar* source_data = input.data;

	for (int y = 0; y < INPUT_HEIGHT; ++y) {
		const uchar* source_row = source_data + (y * INPUT_WIDTH * 3);
		for (int x = 0; x < INPUT_WIDTH; ++x) {
			const uchar* source_pixel = source_row + (x * 3);
			for (int c = 0; c < 3; ++c) {
				const uchar* source_value = source_pixel + c;
				input_tensor_mapped(0, y, x, c) = (*source_value)*1.0/255;
				//std::cout <<"input_tensor_mapped:"<< input_tensor_mapped(0, y, x, c) << std::endl;
			}
	    }
	}
}


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


Status MatToTensorOfFloat_1(Mat input , const int input_height,
		const int input_width, const float input_mean,
		const float input_std, tensorflow::Tensor &out_tensors) {
	// resize
	resize(input, input, Size(input_height, input_width));
	// color convert
	cvtColor(input, input, COLOR_BGR2RGB);

	// allocate a Tensor
	//tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
	// get pointer to memory for that Tensor
	float *p  = out_tensors.flat<float>().data();
	// create a "fake" cv::Mat from it
	Mat outputImg(299, 299, CV_32FC3, p);
	// use it here as a destination
	input.convertTo(outputImg, CV_32FC3);
	// process for image
	outputImg = (outputImg -input_mean)*1.0/input_std;

	//debug
//	for (int y = 0; y < input_height; ++y) {
//		std::cout <<"input_tensor_mapped:"<<  p[y] << std::endl;
//	}

	return Status::OK();
}


Status MatToTensorOfFloat_2(Mat input , const int input_height,
		const int input_width, const float input_mean,
		const float input_std, tensorflow::Tensor &out_tensors) {
	// resize the image
	resize(input, input, Size(input_height, input_width));
	//convert BGR to RGB
	input.convertTo(input, CV_32FC3);
	//tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
	auto input_tensor_mapped = out_tensors.tensor<float, 4>();
	auto start = std::chrono::system_clock::now();
	//Copy all the data over
	for (int y = 0; y < input_height; ++y) {
		const float* source_row = ((float*)input.data) + (y * input_height * 3);
	    for (int x = 0; x < input_width; ++x) {
	    	 const float* source_pixel = source_row + (x * 3);
	         input_tensor_mapped(0, y, x, 0) = (source_pixel[2]-input_mean)/input_std;//R
	         input_tensor_mapped(0, y, x, 1) = (source_pixel[1]-input_mean)/input_std;//G
	         input_tensor_mapped(0, y, x, 2) = (source_pixel[0]-input_mean)/input_std;//B
	         //for debug
	         //std::cout <<"input_tensor_mapped:"<<  input_tensor_mapped(0, y, x, 0) << std::endl;
	     }
	 }
	 auto end = std::chrono::system_clock::now();

	return Status::OK();
}

Status MatToTensorOfFloat_3(Mat input , const int input_height,
		const int input_width, const float input_mean,
		const float input_std, tensorflow::Tensor &out_tensors) {

	  resize(input, input, Size(input_height, input_width));
	  cvtColor(input, input, COLOR_BGR2RGB);
	  //imshow( "Display Image", input );
	  //waitKey(0);

	  //tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, input_height, input_width, 3}));
	  auto input_tensor_mapped = out_tensors.tensor<float, 4>();
	  const uchar* source_data = input.data;
	  //Copy all the data over
	  for (int y = 0; y < INPUT_HEIGHT; ++y) {
	  		const uchar* source_row = source_data + (y * INPUT_WIDTH * 3);
	  		for (int x = 0; x < INPUT_WIDTH; ++x) {
	  			const uchar* source_pixel = source_row + (x * 3);
	  			for (int c = 0; c < 3; ++c) {
	  				const uchar* source_value = source_pixel + c;
	  				input_tensor_mapped(0, y, x, c) = (*source_value-input_mean)*1.0/input_std;
	  			    //std::cout <<"input_tensor_mapped:"<<  input_tensor_mapped(0, y, x, c) << std::endl;
	  			}
	  		}
	  	}

	return Status::OK();
}



Status ReadTensorFromImageFile_by_opencv(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
							   tensorflow::Tensor* out_tensors) {
	Mat image;
	image = imread(file_name); // Read the file
	if(!image.data)  {
	        std::cerr << "Could not open or find the image at " << file_name << std::endl;
	        return tensorflow::errors::NotFound("Could not open or find the image at ", file_name,
                    " not found.");
	}

	resize(image,image,Size(input_height, input_width));
	cvtColor(image, image, COLOR_BGR2RGB);
	image.convertTo(image, CV_32FC3);

	//tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
	auto input_tensor_mapped = out_tensors->tensor<float, 4>();

	const float* source_data = (float*)image.data;

	auto start = std::chrono::system_clock::now();
	for (int y = 0; y < input_height; ++y) {
		const float* source_row = source_data + (y * input_height * 3);
		for (int x = 0; x < input_width; ++x) {
			const float* source_pixel = source_row + (x * 3);
			for (int c = 0; c < 3; ++c) {
				const float* source_value = source_pixel + c;
				input_tensor_mapped(0, y, x, c) = (*source_value-input_mean)*1.0/input_std;
			}
		}
	}
	auto end = std::chrono::system_clock::now();

	return Status::OK();

}


// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
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

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
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


// convert opencv load image data into tensor
Status ReadOpencvfile(const string& file_name, const int input_height,
	const int input_width, const float input_mean,
	const float input_std,
	std::vector<Tensor>* out_tensors) {
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

	string input_name = "file_reader";
	string output_name = "normalized";

	tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
	auto input_tensor_mapped = image_tensor.tensor<float, 4>();

	const float* source_data = NULL;

	for (int y = 0; y < INPUT_HEIGHT; ++y) {
		const float* source_row = source_data + (y * INPUT_WIDTH * 3);
		for (int x = 0; x < INPUT_WIDTH; ++x) {
			const float* source_pixel = source_row + (x * 3);
			for (int c = 0; c < 3; ++c) {
				const float* source_value = source_pixel + c;
				input_tensor_mapped(0, y, x, c) = *source_value;
			}
		}
	}

	auto file_reader =
		tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);
	// Now try to figure out what kind of file it is and decode it.
	const int wanted_channels = 3;
	tensorflow::Output image_reader;
	if (tensorflow::str_util::EndsWith(file_name, ".png")) {
		image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
			DecodePng::Channels(wanted_channels));
	}
	else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
		// gif decoder returns 4-D tensor, remove the first dim
		image_reader = Squeeze(root.WithOpName("squeeze_first_dim"),
			DecodeGif(root.WithOpName("gif_reader"),
				file_reader));
	}
	else {
		// Assume if it's neither a PNG nor a GIF then it must be a JPEG.
		image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
			DecodeJpeg::Channels(wanted_channels));
	}
	// Now cast the image data to float so we can do normal math on it.
	auto float_caster =
		Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
	// The convention for image ops in TensorFlow is that all images are expected
	// to be in batches, so that they're four-dimensional arrays with indices of
	// [batch, height, width, channel]. Because we only have a single image, we
	// have to add a batch dimension of 1 to the start with ExpandDims().
	auto dims_expander = ExpandDims(root, float_caster, 0);

	// Bilinearly resize the image to fit the required dimensions.
	auto resized = ResizeBilinear(
		root, dims_expander,
		Const(root.WithOpName("size"), { input_height, input_width }));
	// Subtract the mean and divide by the scale.
	Div(root.WithOpName(output_name), Sub(root, resized, { input_mean }),
	{ input_std });

	// This runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensor.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({}, { output_name }, {}, out_tensors));
	return Status::OK();
}


// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
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
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      const string& labels_file_name) {
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  }
  return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected) {
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 1;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}
