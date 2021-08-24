//实现halcon中的高斯差分滤波傅里叶变换

//高斯差分滤波器
Mat EnhanceEdgeFilter(Mat &src, const int &std1, const int &std2)
{
	Mat filter1(src.size(), CV_32FC1, cv::Scalar(0));   //标准差 10
	Mat filter2(src.size(), CV_32FC1, cv::Scalar(0));   //标准差 3

	Mat kernel_y = cv::getGaussianKernel(filter1.rows, std1);
	Mat kernel_x = cv::getGaussianKernel(filter1.cols, std1);
	cv::transpose(kernel_x, kernel_x);
	filter1 = kernel_y * kernel_x;
	kernel_y = cv::getGaussianKernel(filter2.rows, std2);
	kernel_x = cv::getGaussianKernel(filter2.cols, std2);
	cv::transpose(kernel_x, kernel_x);
	filter2 = kernel_y * kernel_x;

	Mat TranFrom1, dst;
	cv::Mat planes[] = { cv::Mat_<float>(filter1),cv::Mat::zeros(filter1.size(),CV_32F) };

	cv::Mat complexI;
	cv::merge(planes, 2, complexI);   //将planes中的两个Mat合成一个Mat--complexI, merge与split相对

	cv::dft(complexI, complexI);

	cv::split(complexI, planes);

	TranFrom1 = planes[0];
	cv::normalize(TranFrom1, dst, 0, 1, NORM_MINMAX);


	Mat TranFrom2, dst2;
	cv::Mat planes2[] = { cv::Mat_<float>(filter2),cv::Mat::zeros(filter2.size(),CV_32F) };

	cv::Mat complexI2;
	cv::merge(planes2, 2, complexI2);   //将planes中的两个Mat合成一个Mat--complexI, merge与split相对

	cv::dft(complexI2, complexI2);

	cv::split(complexI2, planes2);

	TranFrom2 = planes2[0];
	cv::normalize(TranFrom2, dst2, 0, 1, NORM_MINMAX);

	Mat dst_filter = dst - dst2;

	return dst_filter;
}


//傅里叶变换
void FFT(const Mat &img, Mat &dst)
{
	cv::Mat src = img.clone();

	cv::Mat planes[] = { cv::Mat_<float>(src),cv::Mat::zeros(src.size(),CV_32F) };

	cv::Mat complexI;
	cv::merge(planes, 2, complexI);   //将planes中的两个Mat合成一个Mat--complexI, merge与split相对

	cv::dft(complexI, complexI);

	cv::split(complexI, planes);  //planes[0] = Re(DFT(I),planes[1] = Im(DFT(I))
								  //即planes[0]为实部,planes[1]为虚部(傅里叶变换后有实部和虚部)	

	cv::Mat imgI = planes[0].clone();
	cv::Mat img2 = planes[1].clone();

	cv::Mat result, blur1, blur2;
	Mat filter = EnhanceEdgeFilter(imgI, 10, 3);

	cv::multiply(imgI, filter, blur1);
	cv::multiply(img2, filter, blur2);
	Mat planes2[] = { blur1,blur2 };
	merge(planes2, 2, result);

	idft(result, result);
	split(result, planes);
	//	magnitude(planes[0], planes[1], planes[0]);

	normalize(planes[0], planes[0], 255, 0, NORM_MINMAX);  //归一化便于显示

	dst = planes[0];

	int cx = dst.cols / 2;
	int cy = dst.rows / 2;

	Mat q0 = dst(cv::Rect(0, 0, cx, cy));
	Mat q1 = dst(cv::Rect(cx, 0, cx, cy));
	Mat q2 = dst(cv::Rect(0, cy, cx, cy));
	Mat q3 = dst(cv::Rect(cx, cy, cx, cy));

	//变换左上角和右下角象限
	cv::Mat temp;
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);

	//变换右上角和左下角象限
	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);

	dst.convertTo(dst, CV_8UC1);
}