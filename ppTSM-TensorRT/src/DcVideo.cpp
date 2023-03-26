#include <windows.h>
#include <fstream>
#include "DcVideo.h"


static bool PathExists(const std::string& path)
{
#ifdef _WIN32
	struct _stat buffer;
	return (_stat(path.c_str(), &buffer) == 0);
#else
	struct stat buffer;
	return (stat(path.c_str(), &buffer) == 0);
#endif  // !_WIN32
}
void print_cost(char* fun, double& startTime)
{
	double endTime = cv::getTickCount();
	double costTime = (endTime - startTime) * 1000 / getTickFrequency();//获取的单位是ms
	printf("%s cost time %f\n", fun, costTime);
}


TSMRecognizer::TSMRecognizer(string& in_engine_path)
{
	isInit = Init(in_engine_path);
}

bool TSMRecognizer::Init(string& in_engine_path)
{
	bool tsmInit = TSMEngine::getInstant()->TSMInit(in_engine_path);
	return tsmInit;
}

int TSMRecognizer::Infer(string& video_path)
{
	//===> SampleFramesFromVideo
	//double t0 = cv::getTickCount();
	std::vector<cv::Mat> sampled_frames;
	int isSample = Utility::SampleFramesFromVideo(video_path, sampled_frames, this->video_param.NUM_SEGMENTS, this->video_param.SEG_LEN);
	//int isSample = Utility::SampleFramesFromVideo(video_path, sampled_frames);
	if (!isSample)
	{
		cout << "open video failed..." << endl;
		return -1;
	}
	//print_cost("SampleFramesFromVideo: ", t0);

	//===> frames_preprocess
	//double t1 = cv::getTickCount();
	const int input_len = video_param.frame_len;
	static float input_data[input_len];
	frames_process.Run(sampled_frames, input_data, 1); //0:单线程， 1: OpenMP, 2: std::thread
	//print_cost("preprocess: ", t1);
	
	//===> Inference
	//double t2 = cv::getTickCount();
	const int output_len = video_param.NUM_CLASSES;
	float prob[output_len];
	bool isInfer = TSMEngine::getInstant()->Inference(input_data, prob);
	if (!isInfer) {
		cout << "Inference fail..." << endl;
		return -1;
	}
	//print_cost("Inference: ", t2);

	//postprocess
	//double t3 = cv::getTickCount();
	std::vector<float> predict_data(video_param.NUM_CLASSES);
	for (int i = 0; i < predict_data.size(); ++i)
	{
		predict_data[i] = prob[i];
		//std::cout << prob[i] << endl;
	}
	
	Softmax::Inplace_Run(predict_data.begin(), predict_data.end());
	int argmax_idx = int(Utility::argmax(predict_data.begin(), predict_data.end()));
	float score = predict_data[argmax_idx];
	//print_cost("postprocess: ", t3);
	//std::cout << video_path << "\tclass: " << argmax_idx << "\tscore: " << score << endl;
	/*if (argmax_idx)
	{
		//MoveFile(video_path.c_str(), target_avi.c_str());
	}*/
	predict_data.clear();
	sampled_frames.clear();

	return argmax_idx;
}