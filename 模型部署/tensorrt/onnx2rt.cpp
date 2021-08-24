#include <iostream>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace nvonnxparser;


class MyLog : public ILogger
{
public:
	void log(Severity severity, const char* msg) 
	{
		//不提示INFO信息，只显示警告和错误
		if (severity != Severity::kINFO)
		{
			std::cout << msg << std::endl;
		}
	}

	MyLog() {}

	~MyLog() {}
};

int maintf() 
{
	samplesCommon::Args args;

	// 1 加载onnx模型
	MyLog gLogger;

	IBuilder* builder = createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

	const char* onnx_filename = "./trtsim.onnx";

	parser->parse(onnx_filename, (size_t)DataType::kFLOAT);
	parser->parseFromFile(onnx_filename, static_cast<int>(Logger::Severity::kWARNING));
	
	for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << parser->getError(i)->desc() << std::endl;
	}
	std::cout << "successfully load the onnx model" << std::endl;

	// 2、build the engine
	unsigned int maxBatchSize = 1;
	builder->setMaxBatchSize(maxBatchSize);
	IBuilderConfig* config = builder->createBuilderConfig();
	size_t MAX_WORKSPACE_SIZE = 1ULL << 30;  //使用1G的内存
	config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	// 3、serialize Model
	IHostMemory *gieModelStream = engine->serialize();
	std::string serialize_str;
	std::ofstream serialize_output_stream;
	serialize_str.resize(gieModelStream->size());
	memcpy((void*)serialize_str.data(), gieModelStream->data(), gieModelStream->size());
	serialize_output_stream.open("./test.trt", ios::binary);
	serialize_output_stream << serialize_str;
	serialize_output_stream.close();

	// 4、deserialize model
	/*MyLog gLogger;
	IRuntime* runtime = createInferRuntime(gLogger);
	std::string cached_path = "./serialize_engine_output.trt";
	std::ifstream fin(cached_path, ios::binary);
	std::string cached_engine = "";
	while (fin.peek() != EOF) {
		std::stringstream buffer;
		buffer << fin.rdbuf();
		cached_engine.append(buffer.str());
	}
	fin.close();
	ICudaEngine* re_engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	if (re_engine == NULL)
	{
		cout << "failed\n";
	}
	else
		cout << "success\n";

	IExecutionContext *context = re_engine->createExecutionContext();*/

	std::cout << "Hello, World!" << std::endl;

	return 0;
}


