#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>


Ort::Session* session_;

int input_h;
int input_w;
std::vector<std::string> input_node_names;
std::vector<std::string> output_node_names;
std::vector<std::string> labels_name;
std::vector<Ort::Value> outputs;
cv::RNG rng;


void modnet_init(const char* module, const char* receive, const char* result, const char* width, const char* height);
void predict_image(std::string src_path, std::string dst_path, int width, int height);


void modnet_init(const char* module, const char* receive, const char* result, const char* width, const char* height)

{
	Ort::Env env;
	Ort::SessionOptions session_options;
	std::wstring w_model_path;
	Ort::AllocatorWithDefaultOptions allocator;

	std::wstringstream wss;
	wss << module;

	env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "mask-rcnn-onnx");
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	w_model_path = wss.str();

	std::cout << "onnxruntime inference try to use CPU Device" << std::endl;

	session_ = new Ort::Session(env, w_model_path.c_str(), session_options);


	int input_nodes_num = static_cast<int>(session_->GetInputCount());
	int output_nodes_num = static_cast<int>(session_->GetOutputCount());

	std::cout << "input_nodes_num : " << input_nodes_num << std::endl;
	std::cout << "output_nodes_num : " << output_nodes_num << std::endl;

	for (int i = 0; i < input_nodes_num; i++) {
		auto input_name = session_->GetInputName(i, allocator);
		input_node_names.push_back(input_name);
	}

	for (int i = 0; i < output_nodes_num; i++) {
		auto output_name = session_->GetOutputName(i, allocator);
		output_node_names.push_back(output_name);
	}

	//旧代码
	//std::wstringstream wss;
	//wss << module;

	//Ort::Env env_;
	//Ort::SessionOptions session_options_;
	//Ort::RunOptions run_options_{ nullptr };
	//Ort::AllocatorWithDefaultOptions allocator;
	//int num_threads = 1;
	//std::vector<int64_t> input_node_dims = { 1, 3, 512, 512 };

	//std::wstring model_path = wss.str();
	std::string src_path(receive);
	std::string dst_path(result);

	//input_node_dims_ = input_node_dims;
	//for (int64_t i : input_node_dims_) {
	//	input_tensor_size_ *= i;
	//	out_tensor_size_ *= i;
	//}

	////std::cout << input_tensor_size_ << std::endl;
	//session_options_.SetIntraOpNumThreads(num_threads);
	//session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	//try {
	//	session_ = Ort::Session(env_, model_path.c_str(), session_options_);
	//}
	//catch (...) {

	//}
	////获取输入name
	//const char* input_name = session_.GetInputName(0, allocator);
	//input_node_names_ = { input_name };
	////std::cout << "input name:" << input_name << std::endl;
	//const char* output_name = session_.GetOutputName(0, allocator);
	//out_node_names_ = { output_name };
	int img_width = std::stoi(width);
	int img_height = std::stoi(height);
	////std::cout << "output name:" << output_name << std::endl;
	predict_image(src_path, dst_path, img_width, img_height);
	std::cout << "modnet init successful" << std::endl;
}


void predict_image(std::string src_path, std::string dst_path, int width, int height)
{
	std::map<int, int> indexArea;
	static int maxarea = 0;
	static int maxindex = 0;
	labels_name = { "person" };
	cv::Mat predict_image;

	cv::Mat image = cv::imread(src_path);
	input_h = image.rows;
	input_w = image.cols;
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(image.cols, image.rows), cv::Scalar(0, 0, 0), true, false);
	std::cout << "preprocess successful" << std::endl;

	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
	size_t tpixels = input_h * input_w * 3;

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	const std::array<const char*, 4> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str(), output_node_names[2].c_str(), output_node_names[3].c_str() };

	try {
		outputs = session_->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	const float* boxes = outputs[0].GetTensorMutableData<float>();
	const int64* labels = outputs[1].GetTensorMutableData<int64>();
	const float* scores = outputs[2].GetTensorMutableData<float>();
	const float* mask_prob = outputs[3].GetTensorMutableData<float>();
	std::cout << "predict_image start" << std::endl;

	auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int rows = outShape[0];

	std::cout << "fixed number: " << rows << std::endl;

	auto mask_tensorShapeInfo = outputs[3].GetTensorTypeAndShapeInfo();
	auto mask_shape = mask_tensorShapeInfo.GetShape();
	std::cout << "mask format: " << mask_shape[0] << "x" << mask_shape[1] << "x" << mask_shape[2] << "x" << mask_shape[3] << std::endl;

	cv::Mat det_output(rows, 4, CV_32F, (float*)boxes);
	for (int i = 0; i < det_output.rows; i++)
	{
		double conf = scores[i];
		int cid = labels[i] - 1;
		// 置信度在0-1之间
		if (conf > 0.85 && cid == 0)
		{
			float x1 = det_output.at<float>(i, 0);
			float y1 = det_output.at<float>(i, 1);
			float x2 = det_output.at<float>(i, 2);
			float y2 = det_output.at<float>(i, 3);

			cv::Rect box;
			box.x = x1;
			box.y = y1;
			box.width = x2 - x1;
			box.height = y2 - y1;

			int img_w1 = width / 3;
			int img_w2 = width / 3 * 2;
			if (img_w1 <= box.x && box.x + box.width <= img_w2)
			{
				int mw = mask_shape[3];
				int mh = mask_shape[2];
				int index = i * mw * mh;
				cv::Mat det_mask(mh, mw, CV_32F, (float*)&mask_prob[index]);
				cv::threshold(det_mask, det_mask, 0.5, 255.0, cv::THRESH_BINARY);
				int maskArea = cv::countNonZero(det_mask);
				//std::cout << "Index: " << i << " " << "Mask area: " << maskArea << std::endl;
				indexArea[i] = maskArea;
			}
		}
	}

	for (const auto& kv : indexArea) {
		std::cout << "键为 " << kv.first << "，对应的值为 " << kv.second << std::endl;
		if (kv.second > maxarea)
		{
			maxarea = kv.second;
			maxindex = kv.first;
		}
	}
	std::cout << "最大面积： " << maxarea << std::endl;
	std::cout << "最大面积index： " << maxindex << std::endl;

	int i = maxindex;
	double conf = scores[i];
	int cid = labels[i] - 1;

	float x1 = det_output.at<float>(i, 0);
	float y1 = det_output.at<float>(i, 1);
	float x2 = det_output.at<float>(i, 2);
	float y2 = det_output.at<float>(i, 3);

	cv::Rect box;
	box.x = x1;
	box.y = y1;
	box.width = x2 - x1;
	box.height = y2 - y1;

	int mw = mask_shape[3];
	int mh = mask_shape[2];
	int index = i * mw * mh;
	cv::Mat det_mask(mh, mw, CV_32F, (float*)&mask_prob[index]);
	cv::threshold(det_mask, det_mask, 0.5, 255.0, cv::THRESH_BINARY);
	//cv::Mat mask, rgb;
	cv::Mat mask;
	det_mask.convertTo(mask, CV_8UC1);
	//cv::Mat rimage = cv::Mat::zeros(mask.size(), mask.type());
	//add(rimage, cv::Scalar(rng.uniform(0, 255)), rimage, mask);
	//cv::Mat gimage = cv::Mat::zeros(mask.size(), mask.type());
	//std::vector<cv::Mat> mlist;
	//mlist.push_back(rimage);
	//mlist.push_back(gimage);
	//mlist.push_back(mask);
	//cv::merge(mlist, rgb);
	//cv::addWeighted(image, 1.0, rgb, 0.5, 0, image);
	//cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8, 0);
	//cv::putText(image, cv::format("%s_%.2f_%d", labels_name[cid].c_str(), conf, i), box.tl(), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2, 8);
	//cv::putText(image, cv::format("FPS"), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

	// 变白色透明背景
	cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
	cv::Mat mask_white(mask.size(), CV_8UC4);

	for (int row = 0; row < mask.rows; row++)
	{
		for (int col = 0; col < mask.cols; col++)
		{
			int pv = mask.at<uchar>(row, col);
			if (pv == 0)
			{
				mask.at<uchar>(row, col) = 255;
				mask_white.at<cv::Vec4b>(row, col)[0] = mask.at<uchar>(row, col);
				mask_white.at<cv::Vec4b>(row, col)[1] = mask.at<uchar>(row, col);
				mask_white.at<cv::Vec4b>(row, col)[2] = mask.at<uchar>(row, col);
				mask_white.at<cv::Vec4b>(row, col)[3] = 0;
				image.at<cv::Vec4b>(row, col)[3] = 0;
			}
			else
			{
				mask.at<uchar>(row, col) = 0;
				mask_white.at<cv::Vec4b>(row, col)[0] = mask.at<uchar>(row, col);
				mask_white.at<cv::Vec4b>(row, col)[1] = mask.at<uchar>(row, col);
				mask_white.at<cv::Vec4b>(row, col)[2] = mask.at<uchar>(row, col);
				mask_white.at<cv::Vec4b>(row, col)[3] = 255;
				image.at<cv::Vec4b>(row, col)[3] = 255;
			}
		}
	}

	cv::add(image, mask_white, predict_image);
	
	cv::imwrite(dst_path, predict_image);
	std::cout << "predict_image successful" << std::endl;
}


int main()
{
	std::cout << "infer...." << std::endl;
	modnet_init("C:\\Users\\hldes\\Desktop\\humanseg\\tmp_file\\mask_rcnn.onnx", "C:\\Users\\hldes\\Desktop\\humanseg\\tmp_file\\111.jpeg", "C:\\Users\\hldes\\Desktop\\humanseg\\tmp_file\\result.png", "640", "480");

	return 0;
}