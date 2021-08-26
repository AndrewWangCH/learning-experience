//仿射变换
//2*3矩阵实现旋转平移, 用Cx, Cy表示旋转中心, α代表角度		getRotationMatrix2D得到的矩阵如下：
//            cosα		sinα	(1-α)*Cx-sinα*Cy
//			  -sinα     cosα    sinα*Cx+(1-cosα)*Cy
//			  前面4个又来旋转后面两个用来平移

//只旋转
void affineRotate(const cv::Mat src, cv::Mat &dst, const cv::Point center, const double angle, const bool mode)
{
	cv::Mat rotateM = cv::getRotationMatrix2D(center, angle, 1.0);  //这里不进行缩放
	
	if (mode)
	{
		double a = sin(angle* CV_PI / 180), b = cos(angle* CV_PI / 180);
		int width = src.size().width;
		int height = src.size().height;
		int width_rotate = int(height * fabs(a) + width * fabs(b)); //旋转后图像宽度   
		int height_rotate = int(width * fabs(a) + height * fabs(b)); //旋转后图像高度   
		rotateM.at<double>(0, 2) += (width_rotate - width) / 2;
		rotateM.at<double>(1, 2) += (height_rotate - height) / 2;
		Mat result = cv::Mat::zeros(cv::Size(width_rotate, height_rotate), CV_8UC1);
		cv::warpAffine(src, result, rotateM, result.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		dst = result.clone();
	}
	else
	{
		cv::warpAffine(src, dst, rotateM, src.size());
	}
}

//只平移
void affineTranslation(const cv::Mat src, cv::Mat &dst, const int dx, const int dy)
{
	//定义平移矩阵
	cv::Mat transM = cv::Mat::zeros(2, 3, CV_32FC1);
	transM.at<float>(0, 0) = 1;
	transM.at<float>(0, 2) = dx; //水平平移量
	transM.at<float>(1, 1) = 1;
	transM.at<float>(1, 2) = dy; //竖直平移量

	cv::warpAffine(src, dst, transM, dst.size());  //平移
}

//平移加旋转
void affineRT(const cv::Mat src, cv::Mat &dst, const cv::Point center, const double angle, const int dx, const int dy)
{
	//旋转矩阵
	cv::Mat rotateM = cv::getRotationMatrix2D(center, angle, 1.0);  //这里不进行缩放
	rotateM.at<double>(0, 2) += dx;
	rotateM.at<double>(1, 2) += dy;
	cv::warpAffine(src, dst, rotateM, src.size());
}


