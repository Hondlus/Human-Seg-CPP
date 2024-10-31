#pragma once
#include "pch.h"
#ifdef HUMANSEG_DLL_EXPORTS
#define HUMANSEG_DLL_API __declspec(dllexport)
#else
#define HUMANSEG_DLL_API __declspec(dllimport)
#endif

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>

Ort::Session session_{ nullptr };

std::vector<Ort::Value> input_tensors_;
std::vector<const char*> input_node_names_;
std::vector<int64_t> input_node_dims_;
size_t input_tensor_size_{ 1 };
std::vector<const char*> out_node_names_;
size_t out_tensor_size_{ 1 };

int image_h;
int image_w;
cv::Mat normalize(cv::Mat& image);
cv::Mat preprocess(cv::Mat image);

HUMANSEG_DLL_API void modnet_init(std::wstring model_path, std::string src_path, std::string dst_path);
cv::Mat predict_image(cv::Mat& src);
void predict_image2(std::string src_path, std::string dst_path);


void modnet_init(std::wstring model_path, std::string src_path, std::string dst_path)
{
	Ort::Env env_;
	Ort::SessionOptions session_options_;
	Ort::RunOptions run_options_{ nullptr };
	Ort::AllocatorWithDefaultOptions allocator;

	int num_threads = 1;
	std::vector<int64_t> input_node_dims = { 1, 3, 512, 512 };

	input_node_dims_ = input_node_dims;
	for (int64_t i : input_node_dims_) {
		input_tensor_size_ *= i;
		out_tensor_size_ *= i;
	}

	session_options_.SetIntraOpNumThreads(num_threads);
	session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	try {
		session_ = Ort::Session(env_, model_path.c_str(), session_options_);
	}
	catch (...) {

	}

	const char* input_name = session_.GetInputName(0, allocator);
	input_node_names_ = { input_name };

	const char* output_name = session_.GetOutputName(0, allocator);
	out_node_names_ = { output_name };

	predict_image2(src_path, dst_path);
}

cv::Mat normalize(cv::Mat& image) {
	std::vector<cv::Mat> channels, normalized_image;
	cv::split(image, channels);

	cv::Mat r, g, b;
	b = channels.at(0);
	g = channels.at(1);
	r = channels.at(2);
	b = (b / 255. - 0.5) / 0.5;
	g = (g / 255. - 0.5) / 0.5;
	r = (r / 255. - 0.5) / 0.5;

	normalized_image.push_back(r);
	normalized_image.push_back(g);
	normalized_image.push_back(b);

	cv::Mat out = cv::Mat(image.rows, image.cols, CV_32F);
	cv::merge(normalized_image, out);
	return out;
}

cv::Mat preprocess(cv::Mat image) {
	image_h = image.rows;
	image_w = image.cols;
	cv::Mat dst, dst_float, normalized_image;
	cv::resize(image, dst, cv::Size(int(input_node_dims_[3]), int(input_node_dims_[2])), 0, 0);
	dst.convertTo(dst_float, CV_32F);
	normalized_image = normalize(dst_float);

	return normalized_image;
}

cv::Mat predict_image(cv::Mat& src) {

	cv::Mat preprocessed_image = preprocess(src);
	cv::Mat blob = cv::dnn::blobFromImage(preprocessed_image, 1, cv::Size(int(input_node_dims_[3]), int(input_node_dims_[2])), cv::Scalar(0, 0, 0), false, true);

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	input_tensors_.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_node_dims_.data(), input_node_dims_.size()));

	std::vector<Ort::Value> output_tensors_ = session_.Run(
		Ort::RunOptions{ nullptr },
		input_node_names_.data(),
		input_tensors_.data(),
		input_node_names_.size(),
		out_node_names_.data(),
		out_node_names_.size()
	);
	float* floatarr = output_tensors_[0].GetTensorMutableData<float>();

	cv::Mat mask = cv::Mat::zeros(static_cast<int>(input_node_dims_[2]), static_cast<int>(input_node_dims_[3]), CV_8UC1);

	for (int i{ 0 }; i < static_cast<int>(input_node_dims_[2]); i++) {
		for (int j{ 0 }; j < static_cast<int>(input_node_dims_[3]); ++j) {
			mask.at<uchar>(i, j) = static_cast<uchar>(floatarr[i * static_cast<int>(input_node_dims_[3]) + j] > 0.5);
		}
	}
	cv::resize(mask, mask, cv::Size(image_w, image_h), 0, 0);
	input_tensors_.clear();

	cv::Mat mask_white(mask.size(), CV_8UC3);
	for (int row = 0; row < mask.rows; row++)
	{
		for (int col = 0; col < mask.cols; col++)
		{
			int pv = mask.at<uchar>(row, col);
			if (pv == 0)
			{
				mask.at<uchar>(row, col) = 255;
			}
			else
			{
				mask.at<uchar>(row, col) = 0;
			}
			mask_white.at<cv::Vec3b>(row, col)[0] = mask.at<uchar>(row, col);
			mask_white.at<cv::Vec3b>(row, col)[1] = mask.at<uchar>(row, col);
			mask_white.at<cv::Vec3b>(row, col)[2] = mask.at<uchar>(row, col);
		}
	}
	return mask, mask_white;
}

void predict_image2(std::string src_path, std::string dst_path) {
	cv::Mat image = cv::imread(src_path);
	cv::Mat mask, mask_white = predict_image(image);
	cv::Mat predict_image;

	cv::add(image, mask_white, predict_image);

	cv::imwrite(dst_path, predict_image);
}