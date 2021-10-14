void calCannyThreshold(cv::Mat &ImgG, int &low, int &high)
{
	Mat ImgT, dx, dy, grad;
	cv::resize(ImgG, ImgT, cv::Size(ImgG.cols / 10, ImgG.rows / 10));
	cv::Sobel(ImgT, dx, CV_16SC1, 1, 0);
	cv::Sobel(ImgT, dy, CV_16SC1, 0, 1);
	short *_dx = (short*)dx.data, *_dy = (short*)dy.data;

	int subpixel_num = dx.rows*dx.cols;
	grad.create(1, subpixel_num, CV_32SC1);
	int* _grad = (int*)grad.data;
	int maxGrad(0);
	for (int i = 0; i < subpixel_num; i++)
	{
		_grad[i] = std::abs(_dx[i]) + std::abs(_dy[i]);
		if (maxGrad < _grad[i])
			maxGrad = _grad[i];
	}

	//set magic numbers
	const int NUM_BINS = 64;
	const double percent_of_pixels_not_edges = 0.7;
	const double threshold_ratio = 0.4;
	int bins[NUM_BINS] = { 0 };


	//compute histogram
	int bin_size = std::floorf(maxGrad / float(NUM_BINS) + 0.5f) + 1;
	if (bin_size < 1) bin_size  = 1;
	for (int i = 0; i < subpixel_num; i++)
	{
		bins[_grad[i] / bin_size]++;
	}

	//% Select the thresholds
	float total(0.f);
	float target = float(subpixel_num * percent_of_pixels_not_edges);

	high = 0;
	while (total < target)
	{
		total += bins[high];
		high++;
	}
	high *= (255.0f / NUM_BINS);
	low = threshold_ratio*float(high);
}