std::vector<cv::String> GetOutputsNames(cv::dnn::Net &net)
{
	static std::vector<cv::String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		std::vector<cv::String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}


// openvino模型预测代码
int predect()
{
	cv::dnn::Net net = cv::dnn::readNetFromModelOptimizer("xml", "bin");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	std::vector<cv::String> names = GetOutputsNames(net);


	Mat img = cv::imread("image", 0);
	cv::resize(img, img, Size(128, 160));
	Mat input(img.size(), CV_32FC1, Scalar(0));
	cv::dnn::blobFromImage(img, input, 1 / 255.0, cv::Size(128, 160), cv::Scalar(0), true, false);
	net.setInput(input);
	std::vector<cv::Mat> outs;
	net.forward(outs, names);

	if (outs.size() <= 0)
	{
		return -1;
	}

	cv::Point max_pt(0, 0);
	cv::minMaxLoc(outs[0], NULL, NULL, NULL, &max_pt);

	int max_index = max_pt.x;

	cout << max_index << endl;

	return 0;
}