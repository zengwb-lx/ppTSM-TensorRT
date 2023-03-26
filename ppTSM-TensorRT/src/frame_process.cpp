#include "frame_process.h"
#include "DcVideo.h"
#include <algorithm>
#include <thread>

static void preprocess2(float* inputTensorValues, const cv::Mat& srcframe)
{
	//=====================================================================
	std::vector<float> mean_ = { 0.485f, 0.456f, 0.406f };
	std::vector<float> std_vals = { 0.229f, 0.224f, 0.225f };
	cv::Size input_WH(750, 500);

	cv::Mat img = srcframe.clone();
	cv::Mat resize_img, imgRGBresize, img_float;
	cv::resize(img, resize_img, input_WH, 0.0f, 0.0f, cv::INTER_LINEAR);

	cvtColor(resize_img, imgRGBresize, cv::COLOR_BGR2RGB);	//转RGB
	imgRGBresize.convertTo(img_float, CV_32F, 1.0 / 255);  //divided by 255转float								   
	std::vector <cv::Mat> channels(3); //cv::Mat channels[3]; //分离通道进行HWC->CHW
	cv::Mat dst;
	cv::split(img_float, channels);

#pragma omp parallel for
	for (int i = 0; i < img_float.channels(); i++)	//标准化ImageNet
	{
		channels[i] -= mean_[i];  // mean均值
		channels[i] /= std_vals[i];   // std方差
	}
	cv::merge(channels, dst);
	int img_float_len = img_float.cols * img_float.rows;

#pragma omp parallel for
	for (int i = 0; i<img_float.rows; i++)
	{
		float* pixel = dst.ptr<float>(i);
		for (int j = 0; j<img_float.cols; j++)
		{
			inputTensorValues[i * img_float.cols + j] = pixel[0];
			inputTensorValues[1 * img_float_len + i * img_float.cols + j] = pixel[1];
			inputTensorValues[2 * img_float_len + i * img_float.cols + j] = pixel[2];
			pixel += 3;
		}
	}
}


FrameProcess::FrameProcess()
{
	this->NUM_SEGMENTS = this->video_param.NUM_SEGMENTS;
	this->scale_size = this->video_param.SCALE_SIZE;
	this->input_WH = cv::Size(this->video_param.INPUT_W, this->video_param.INPUT_H);
}
FrameProcess::~FrameProcess()
{

}

void FrameProcess::preprocess(const cv::Mat& srcframe, float* inputTensorValues)
{
	//=====================================================================
	cv::Mat img = srcframe.clone();
	cv::Mat resize_img, imgRGBresize, img_float;
	cv::resize(img, resize_img, this->input_WH, 0.0f, 0.0f, cv::INTER_LINEAR);

	cvtColor(resize_img, imgRGBresize, cv::COLOR_BGR2RGB);	//转RGB
	imgRGBresize.convertTo(img_float, CV_32F, 1.0 / 255);  //divided by 255转float

	std::vector<cv::Mat> channels(3); //cv::Mat channels[3]; 分离通道进行HWC->CHW
	cv::Mat dst;
	cv::split(img_float, channels);
	//#pragma omp parallel for
	for (int i = 0; i < img_float.channels(); i++)	//标准化ImageNet
	{
		channels[i] -= this->mean_[i];  // mean均值
		channels[i] /= this->std_vals[i];   // std方差
	}
	cv::merge(channels, dst);
	int img_float_len = img_float.cols * img_float.rows;
	//#pragma omp parallel for
	for (int i = 0; i<img_float.rows; i++)
	{
		float* pixel = dst.ptr<float>(i);
		for (int j = 0; j<img_float.cols; j++)
		{
			inputTensorValues[i * img_float.cols + j] = pixel[0];
			inputTensorValues[1 * img_float_len + i * img_float.cols + j] = pixel[1];
			inputTensorValues[2 * img_float_len + i * img_float.cols + j] = pixel[2];
			pixel += 3;
		}
	}
}
void FrameProcess::preprocess3(const cv::Mat& srcframe, float* inputTensorValues)
{
	//=====================================================================
	/*std::vector<float> mean_ = { 0.485f, 0.456f, 0.406f };
	std::vector<float> std_vals = { 0.229f, 0.224f, 0.225f };
	cv::Size input_WH(750, 500);*/

	cv::Mat img = srcframe.clone();
	cv::Mat resize_img, imgRGBresize, img_float;
	cv::resize(img, resize_img, input_WH, 0.0f, 0.0f, cv::INTER_LINEAR);

	cvtColor(resize_img, imgRGBresize, cv::COLOR_BGR2RGB);	//转RGB
	imgRGBresize.convertTo(img_float, CV_32F, 1.0 / 255);  //divided by 255转float								   
	std::vector <cv::Mat> channels(3); //cv::Mat channels[3]; //分离通道进行HWC->CHW
	cv::Mat dst;
	cv::split(img_float, channels);

#pragma omp parallel for
	for (int i = 0; i < img_float.channels(); i++)	//标准化ImageNet
	{
		channels[i] -= mean_[i];  // mean均值
		channels[i] /= std_vals[i];   // std方差
	}
	cv::merge(channels, dst);
	int img_float_len = img_float.cols * img_float.rows;

#pragma omp parallel for
	for (int i = 0; i<img_float.rows; i++)
	{
		float* pixel = dst.ptr<float>(i);
		for (int j = 0; j<img_float.cols; j++)
		{
			inputTensorValues[i * img_float.cols + j] = pixel[0];
			inputTensorValues[1 * img_float_len + i * img_float.cols + j] = pixel[1];
			inputTensorValues[2 * img_float_len + i * img_float.cols + j] = pixel[2];
			pixel += 3;
		}
	}
}

//method --> default:单线程， 1: OpenMP, 2: std::thread
void FrameProcess::Run(const std::vector<cv::Mat> &sampled_frames, float* input_frames, int method)
{
	const int image_len = input_WH.height * input_WH.width * 3;
	const int num_frame = sampled_frames.size();
	std::vector<float*> vec_input_data(num_frame);
	switch (method)
	{
	case 1:
	{
		for (int i = 0; i < num_frame; ++i)
		{
			vec_input_data[i] = new float[image_len];
		}
		// Preprocess onnx method 
		double t1 = cv::getTickCount();
		#pragma omp parallel for
		for (int i = 0; i < num_frame; ++i)
		{
			cv::Mat frame_i = sampled_frames[i].clone();
			if (!frame_i.empty())
			{
				preprocess3(sampled_frames[i], vec_input_data[i]);
			}
		}
		//print_cost("rrrrrr: ", t1);
		double t2 = cv::getTickCount();
		#pragma omp parallel for
		for (int i = 0; i < vec_input_data.size(); ++i)
		{
			for (int j = 0; j < image_len; ++j)
			{
				input_frames[i * image_len + j] = vec_input_data[i][j];
			}
			delete[] vec_input_data[i];
		}
		break;
	}
	case 2:
	{
		for (int i = 0; i < num_frame; ++i)
		{
			vec_input_data[i] = new float[image_len];
		}
		std::vector<std::thread> threads;
		// Preprocess onnx method
		for (int i = 0; i < num_frame; ++i)
		{
			const cv::Mat frame_i = sampled_frames[i];
			if (!frame_i.empty())
			{
				threads.push_back(std::thread(&preprocess2, std::ref(vec_input_data[i]), sampled_frames[i]));
			}
		}
		for (auto& t : threads)
		{
			t.join();
		}
		#pragma omp parallel for
		for (int i = 0; i < vec_input_data.size(); ++i)
		{
			for (int j = 0; j < image_len; ++j)
			{
				input_frames[i * image_len + j] = vec_input_data[i][j];
			}
			delete[] vec_input_data[i];
		}
		break;
	}
	default:
	{
		for (int i = 0; i < num_frame; ++i)
		{
			vec_input_data[i] = new float[image_len];
		}
		//#pragma omp parallel for
		for (int i = 0; i < num_frame; ++i)
		{
			cv::Mat frame_i = sampled_frames[i].clone();
			if (!frame_i.empty())
			{
				this->preprocess(sampled_frames[i], vec_input_data[i]);
			}
		}
		//#pragma omp parallel for
		for (int i = 0; i < vec_input_data.size(); ++i)
		{
			for (int j = 0; j < image_len; ++j)
			{
				//float p = vec_input_data[i][j];
				input_frames[i * image_len + j] = vec_input_data[i][j];
			}
			delete[] vec_input_data[i];
		}
		break;
	}
	}
}

bool Utility::SampleFramesFromVideo(const std::string &VideoPath, std::vector<cv::Mat>& sampled_frames, const int &num_seg, const int &seg_len)
{
	//double t1 = cv::getTickCount();
	cv::VideoCapture capture(VideoPath); // Create a video object
	if (!capture.isOpened())
	{
		printf("[Error] video cannot be opened, please check the video [%s]\n", VideoPath.c_str());
		capture.release();
		return false;
	}

	int frames_len = capture.get(cv::CAP_PROP_FRAME_COUNT); // Get the total number of video frames

	std::vector<int> frames_idx;
	if (frames_len == 60)
	{
		frames_idx = { 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46 };
	}
	else
	{
		int average_dur = int(frames_len / num_seg);
		for (int i = 0; i < num_seg; ++i)
		{
			int idx = 0;
			if (average_dur >= seg_len)
			{
				idx = (average_dur - 1) / 2;
				idx += i * average_dur;
			}
			else if (average_dur >= 1)
			{
				idx += i * average_dur;
			}
			else
			{
				idx = i;
			}
			for (int j = idx; j < idx + seg_len; ++j)
			{
				frames_idx.emplace_back(j % frames_len);
			}
		}
	}
	const int num_frame = frames_idx.size();
	sampled_frames.resize(num_frame);
	if (num_frame != num_seg)
	{
		return false;
	}
	cv::Mat frame;
	int i = 0;
	int n = 0;
	while (capture.isOpened())
	{
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		if (std::find(frames_idx.begin(), frames_idx.end(), i) != frames_idx.end())
		{
			frame.copyTo(sampled_frames[n]);
			n += 1;
			//sampled_frames.push_back(frame);
			//frames_idx.erase(frames_idx.begin());
		}
		if (i > frames_idx[num_frame - 1]) {
			break;
		}
		i += 1;
	}
	//print_cost("Sample: ", t1);
	capture.release(); // Release the video object
	return true;
}

bool Utility::SampleFramesFromVideo(const std::string &VideoPath, std::vector<cv::Mat>& sampled_frames)
{
	//double t1 = cv::getTickCount();
	cv::VideoCapture capture(VideoPath); // Create a video object
	if (!capture.isOpened())
	{
		//printf("[Error] video cannot be opened, please check the video [%s]\n", VideoPath.c_str());
		capture.release();
		return false;
	}
	std::vector<int> frames_idx = { 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46 };
	const int num_frame = frames_idx.size();
	sampled_frames.resize(num_frame);

	//int frames_len = capture.get(cv::CAP_PROP_FRAME_COUNT); // Get the total number of video frames
	cv::Mat frame;
	int i = 0;
	int n = 0;
	while (capture.isOpened())
	{
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		if (std::find(frames_idx.begin(), frames_idx.end(), i) != frames_idx.end())
		{
			frame.copyTo(sampled_frames[n]);
			n += 1;
			//sampled_frames.push_back(frame);
			//frames_idx.erase(frames_idx.begin());
		}
		if (i > frames_idx[num_frame - 1]) {
			break;
		}
		i += 1;
	}
	//print_cost("Sample: ", t1);
	capture.release(); // Release the video object
	return true;
}

