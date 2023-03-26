#pragma once
#include <iostream>
#include <vector>
#include <io.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "frame_process.h"
#include "postprocess_op.h"
//tensorrt
#include "trt_engine.h"

using namespace std;
using namespace cv;

void print_cost(char* fun, double& startTime);

class TSMRecognizer
{
public:
	TSMRecognizer(string& engine_path);
	bool isInit = false;
	int Infer(string& video_path);
private:
	VideoParam video_param;
	FrameProcess frames_process;
	bool Init(string& in_engine_path);

};
