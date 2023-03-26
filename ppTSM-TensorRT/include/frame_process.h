#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include "trt_engine.h"

class FrameProcess
{
public:
	FrameProcess();
	~FrameProcess();
	void Run(const std::vector<cv::Mat> &sampled_frames, float* input_frames, int method);
private:
	VideoParam video_param;
	std::vector<float> mean_ = { 0.485f, 0.456f, 0.406f };
	std::vector<float> std_vals = { 0.229f, 0.224f, 0.225f };
	int NUM_SEGMENTS;
	int scale_size;
	cv::Size input_WH;

	void preprocess(const cv::Mat& srcframe, float* inputTensorValues);
	void preprocess3(const cv::Mat& srcframe, float* inputTensorValues);

};// class FrameProcess

class Utility
{
public:
	template <class ForwardIterator> inline static size_t argmax(ForwardIterator first, ForwardIterator last)
	{
		return std::distance(first, std::max_element(first, last));
	}

	static bool SampleFramesFromVideo(const std::string &VideoPath, std::vector<cv::Mat>& sampled_frames, const int &num_seg, const int &seg_len);
	static bool SampleFramesFromVideo(const std::string &VideoPath, std::vector<cv::Mat>& sampled_frames);
};
