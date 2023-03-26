#include "trt_engine.h"

bool CHECK_CUDA(cudaError_t status)
{
	bool ret = status == 0 ? true: false;
	if (!ret) {
		std::cout << "[ppTSM] CHECK_CUDA Error" << std::endl;
	}
	return ret;
}
TSMEngine::TSMEngine()
{

}

TSMEngine* TSMEngine::m_instant = NULL;

TSMEngine* TSMEngine::getInstant()
{
	if (m_instant == NULL) {
		m_instant = new TSMEngine();
	}
	return m_instant;
}

TSMEngine::~TSMEngine()
{
	// Release stream and buffers
	cudaFree(buffers[inputIndex]);
	cudaFree(buffers[outputIndex]);
	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
}

// trtexec.exe --onnx=ppTSM.onnx --saveEngine=ppTSM8218.engine --workspace=1024 --fp16
bool TSMEngine::TSMInit(std::string& model_path)
{
	this->INPUT_H = video_param.INPUT_H;
	this->INPUT_W = video_param.INPUT_W;
	this->OUTPUT_SIZE = video_param.OUTPUT_SIZE;
	this->NUM_SEGMENTS = video_param.NUM_SEGMENTS;

	//this->ENGINE_PATH = video_param.ENGINE_PATH;
	this->INPUT_BLOB_NAME = video_param.INPUT_BLOB_NAME;
	this->OUTPUT_BLOB_NAME = video_param.OUTPUT_BLOB_NAME;

	cudaSetDevice(0);
	// deserialize the .engine
	std::ifstream file(model_path, std::ios::binary);
	if (!file.good()) {
		//std::cout << "[ppTSM] file is not good" << std::endl;
		return false;
	}
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	trtModelStream = new char[size];
	//assert(trtModelStream);
	file.read(trtModelStream, size);
	file.close();

	static Logger gLogger;
	runtime = createInferRuntime(gLogger);
	if (runtime == nullptr) { return false; }
	engine = runtime->deserializeCudaEngine(trtModelStream, size);
	if (engine == nullptr) 
	{ 
		std::cout << "[ppTSM] engine is  nullptr" << std::endl;
		return false; 
	}
	if ((int)engine->getNbBindings() != 2) { return false; }
	context = engine->createExecutionContext();
	if (context == nullptr) { return false; }

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
	if (inputIndex != 0 || outputIndex != 1) 
	{
		return false;
	}
	// Create GPU buffers on device
	if (!CHECK_CUDA(cudaMalloc(&buffers[inputIndex], BatchSize * NUM_SEGMENTS * 3 * INPUT_H * INPUT_W * sizeof(float)))) {
		return false;
	}
	if (!CHECK_CUDA(cudaMalloc(&buffers[outputIndex], BatchSize * OUTPUT_SIZE * sizeof(float)))) {
		return false;
	}

	delete[] trtModelStream;
	return true;
}

bool TSMEngine::doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// Create stream
	if (!CHECK_CUDA(cudaStreamCreate(&stream))) {
		return false;
	}
	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	if (!CHECK_CUDA(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * NUM_SEGMENTS * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream))) {
		return false;
	}
	context.enqueue(batchSize, buffers, stream, nullptr);
	if (!CHECK_CUDA(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream))) {
		return false;
	}
	cudaStreamSynchronize(stream);

	// Release stream
	cudaStreamDestroy(stream);

	return true;
}

bool TSMEngine::Inference(float *data, float *prob)
{
	if (!doInference(*context, data, prob, 1)) {
		std::cout << "[ppTSM] doInference Error" << std::endl;
		return false;
	}
	return true;
}