#pragma once
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

using namespace nvinfer1;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

bool CHECK_CUDA(cudaError_t status);

struct VideoParam
{
	static const int NUM_CLASSES = 400;
	static const int INPUT_H = 224;
	static const int INPUT_W = 224;
	static const int OUTPUT_SIZE = 400;
	static const int NUM_SEGMENTS = 16;
	static const int SEG_LEN = 1;
	static const int SCALE_SIZE = 224;
	static const int frame_len = NUM_SEGMENTS * INPUT_H * INPUT_W * 3;

	//char* ENGINE_PATH;
	char* INPUT_BLOB_NAME = "data_batch_0";
	char* OUTPUT_BLOB_NAME = "linear_1.tmp_1";
};

class TSMEngine
{
public:
	TSMEngine();
	~TSMEngine();
	static TSMEngine* getInstant();
	bool TSMEngine::doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	bool TSMEngine::Inference(float *data, float *prob);
	bool TSMInit(std::string& model_path);
private:
	static TSMEngine* m_instant;
	char* ENGINE_PATH;
	int INPUT_H = 224;
	int INPUT_W = 224;
	int OUTPUT_SIZE = 400;
	int NUM_SEGMENTS = 16;
	VideoParam video_param;

	const int BatchSize = 1;
	char* INPUT_BLOB_NAME = "data_batch_0";
	char* OUTPUT_BLOB_NAME = "linear_1.tmp_1";
	int inputIndex;
	int outputIndex;
	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	void* buffers[2];

	int DEVICE = 0;
	char *trtModelStream = nullptr;
	cudaStream_t stream;
	IRuntime* runtime;
	ICudaEngine* engine;
	IExecutionContext* context;
};