#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <logger.h>
#include <common.h>
#include <cuda_runtime_api.h>
#include <time.h>

#include <io.h>


using namespace nvinfer1;
using namespace plugin;

namespace {

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

	void Mat2MyTensor(cv::Mat &img, cv::Mat &tensor)
	{
		tensor = img.clone();
		tensor.convertTo(tensor, CV_32FC1);
		tensor = tensor / 255;
	}

	int ReadImgName(const std::string &img_path, const std::string &img_name, std::vector<std::string> &vec)
	{
		intptr_t handle;
		struct _finddata_t fileinfo;
		handle = _findfirst(img_path.c_str(), &fileinfo);
		do
		{
			std::string name = img_name + fileinfo.name;
			vec.push_back(name);
		} while (!_findnext(handle, &fileinfo));
		int num = vec.size();
		return num;
	}
}


int mainv3()
{
	int TP = 0, FP = 0, FN = 0, TN = 0;

	static const int MODEL_CHANNEL = 1;
	static const int MODEL_WIDTH = 512;
	static const int MODEL_HEIGHT = 512;
	static const int MODEL_NUM_CLASS = 2;
	const int BatchSize = 1;

	std::string trt_path = "./v6.2_cls.trt";
	MyLog _log;

	IRuntime* runtime = createInferRuntime(_log);

	std::ifstream fin(trt_path, std::ios_base::in | std::ios_base::binary);

	// 1.将文件中的内容读取至cached_engine字符串
	std::string modelData = "";
	while (fin.peek() != EOF)
	{
		// 使用fin.peek()防止文件读取时无限循环
		std::stringstream buffer;
		buffer << fin.rdbuf();
		modelData.append(buffer.str());
	}
	fin.close();

	// 2.将序列化得到的结果进行反序列化，以执行后续的inference
	ICudaEngine* engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size(), nullptr);

	//inference推断过程
	IExecutionContext *context = engine->createExecutionContext();

	int Input1 = engine->getBindingIndex("data1");  //src
	int Input2 = engine->getBindingIndex("data2");  //fft
	int Output1 = engine->getBindingIndex("seg");
	int Output2 = engine->getBindingIndex("cls");

	//申请GPU显存
	void* buffers[4] = { NULL, NULL, NULL, NULL };

	size_t pitch = MODEL_WIDTH * sizeof(float);

	cudaMallocPitch(&buffers[Input1], &pitch, sizeof(float) * MODEL_WIDTH, MODEL_HEIGHT);  //src
	cudaMallocPitch(&buffers[Input2], &pitch, sizeof(float) * MODEL_WIDTH, MODEL_HEIGHT); //fft
	cudaMallocPitch(&buffers[Output1], &pitch, sizeof(float) * MODEL_WIDTH, MODEL_HEIGHT); //seg
	cudaMalloc(&buffers[Output2], MODEL_NUM_CLASS * sizeof(float)); //cls
	//cudaMallocPitch(&buffers[Output2], &pitch, sizeof(float) * BatchSize, MODEL_NUM_CLASS); //cls

	std::string src_img_path = "./train/src/*.jpg";
	std::string src_img_name = "./train/src/";
	std::vector<std::string>src_img_vec;

	std::string fft_img_path = "./train/fft_src/*.jpg";
	std::string fft_img_name = "./train/fft_src/";
	std::vector<std::string>fft_img_vec;
	int img_num = ReadImgName(src_img_path, src_img_name, src_img_vec);
	ReadImgName(fft_img_path, fft_img_name, fft_img_vec);

	int count = 0;
	DWORD start_time = GetTickCount();
	for (int index = 0; index < img_num; ++index)
	{
		std::string str = src_img_vec[index];
		int start_index = str.find_last_of("/");
		std::string label = str.substr(start_index + 1, 2);

		//读取图片，并进行转换
		cv::Mat src_img = cv::imread(src_img_vec[index], 0);
		cv::Mat fft_img = cv::imread(fft_img_vec[index], 0);
		cv::Mat src_tensor, fft_tensor;
		Mat2MyTensor(src_img, src_tensor);
		Mat2MyTensor(fft_img, fft_tensor);


		//torch N H W C --> tensorrt N C H W   在内存中操作
		void *data_src = malloc(BatchSize * MODEL_CHANNEL * MODEL_WIDTH * sizeof(float)* MODEL_HEIGHT);
		void *data_fft = malloc(BatchSize * MODEL_CHANNEL * MODEL_HEIGHT * MODEL_WIDTH * sizeof(float));

		/*memcpy(data_src, src_tensor.ptr<unsigned char>(0), MODEL_WIDTH * MODEL_HEIGHT * sizeof(float));
		memcpy(data_fft, fft_tensor.ptr<unsigned char>(0), MODEL_WIDTH * MODEL_HEIGHT * sizeof(float));*/
		memcpy(data_src, src_tensor.ptr<float>(0), MODEL_WIDTH * MODEL_HEIGHT * sizeof(float));
		memcpy(data_fft, fft_tensor.ptr<float>(0), MODEL_WIDTH * MODEL_HEIGHT * sizeof(float));

		cudaMemcpy2D(buffers[Input1], pitch, data_src, pitch, sizeof(float) * MODEL_WIDTH, MODEL_HEIGHT, cudaMemcpyHostToDevice);
		cudaMemcpy2D(buffers[Input2], pitch, data_fft, pitch, sizeof(float) * MODEL_WIDTH, MODEL_HEIGHT, cudaMemcpyHostToDevice);

		// 5.启动cuda核计算
		//context->enqueueV2(buffers, stream, nullptr);  //异步执行
		context->executeV2(buffers);  //同步执行

		float cls[BatchSize * MODEL_NUM_CLASS] = { 0 };
		static float seg[MODEL_WIDTH * MODEL_HEIGHT] = { 255 };

		//cudaMemcpy(cls, buffers[Output2], MODEL_NUM_CLASS * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(seg, buffers[Output1], MODEL_WIDTH*MODEL_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

		//cudaMemcpy2D(seg, 0, buffers[Output1], pitch, 512, 512 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(cls, buffers[Output2], BatchSize * MODEL_NUM_CLASS * sizeof(float), cudaMemcpyDeviceToHost);

		std::cout << "label: " << label << "  ";
		int label_flag = 0;
		if (label == "OK")
		{
			label_flag = 1;
		}
		else
		{
			label_flag = 0;
		}

		int predict_flag = 0;
		if (cls[0] > cls[1])
			std::cout << "predict: NG  ";
		else
		{
			std::cout << "predict: OK  ";
			predict_flag = 1;
		}

		if (predict_flag == label_flag)
		{
			++count;
		}

		for (int i = 0; i < 2; ++i)
		{
			std::cout << cls[i] << " ";
		}

		cv::Mat ret_seg(cv::Size(512, 512), CV_32FC1);
		std::cout << std::endl;
		for (int i = 0; i < 512; ++i)
		{
			for (int j = 0; j < 512; j++)
			{
				ret_seg.at<float>(j, i) = seg[i + j * 512] * 255;
			}
		}
		ret_seg.convertTo(ret_seg, CV_8UC1);

		if (label_flag == 1)
		{
			if (predict_flag == 1)
			{
				TP++;
			}
			else
				FN++;
		}
		else
		{
			if (predict_flag == 0)
				TN++;
			else
				FP++;
		}
	}
	DWORD end_time = GetTickCount();
	std::cout << "TP = " << TP << " FP = " << FP << " FN = " << FN << " TN = " << TN << std::endl;
	std::cout << "The run time is:" << (end_time - start_time)*1.0 / img_num << "ms!" << std::endl;

	CHECK(cudaFree(buffers[Input1]));
	CHECK(cudaFree(buffers[Input2]));
	CHECK(cudaFree(buffers[Output1]));
	CHECK(cudaFree(buffers[Output2]));

	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "acc = " << count * 1.0 / img_num << std::endl;

	return 0;
}


