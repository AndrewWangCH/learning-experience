//ChanVese模型，主动轮廓模型算法，利用曲线逼近的思想去找边缘
cv::Mat kernelCV(cv::Mat &iniLSF, cv::Mat src, double nu, int mu, int epison, float step)
{
	src.convertTo(src, CV_32FC1);
	cv::Mat Drc = (epison / CV_PI) / (epison*epison + iniLSF.mul(iniLSF));
	cv::Mat arctanImg(iniLSF.size(), CV_32FC1);
	for (int r = 0; r < iniLSF.rows; ++r)
	{
		for (int c = 0; c < iniLSF.cols; ++c)
		{
			float pixVal = iniLSF.at<float>(r, c);
			arctanImg.at<float>(r, c) = atan(pixVal / epison);
		}
	}
	cv::Mat Hea = 0.5*(1 + (2 / CV_PI)*arctanImg);

	Mat gx, gy;
	cv::Sobel(iniLSF, gx, CV_32FC1, 1, 0, 3);
	cv::Sobel(iniLSF, gy, CV_32FC1, 0, 1, 3);
	Mat magImg;
	cv::magnitude(gx, gy, magImg);
	Mat Nx = gx / (magImg + 0.0000001);
	Mat Ny = gy / (magImg + 0.0000001);

	Mat Mxx, Nxx;
	cv::Sobel(Nx, Mxx, CV_32FC1, 1, 0, 3);
	cv::Sobel(Nx, Nxx, CV_32FC1, 0, 1, 3);

	Mat Nyy, Myy;
	cv::Sobel(Ny, Nyy, CV_32FC1, 1, 0, 3);
	cv::Sobel(Ny, Myy, CV_32FC1, 0, 1, 3);

	Mat cur = Nxx + Nyy;

	cv::Mat Length = nu * Drc.mul(cur);
	cv::Mat area = mu * Drc;

	cv::Mat s1 = Hea.mul(src);
	cv::Mat s2 = (1 - Hea).mul(src);
	cv::Mat s3 = 1 - Hea;
	double C1 = cv::sum(s1)[0] / cv::sum(Hea)[0];  //计算s1的像素值和
	double C2 = cv::sum(s2)[0] / cv::sum(s3)[0];

	cv::Mat CVterm = Drc.mul((-1 * (src - C1).mul((src - C1)) + 1 * (src - C2).mul((src - C2))));
	iniLSF = iniLSF + step * (Length + area + CVterm);
	return iniLSF;
}

void ChanVese(const cv::Mat src, std::vector<std::vector<cv::Point>> &contours)
{
	int imgW = src.cols;
	int imgH = src.rows;
	Mat iniLSF = cv::Mat(src.size(), CV_32FC1, Scalar(-1));
	Rect mask = cv::Rect(imgW*0.2, imgH*0.2, imgW*0.3, imgH*0.3);
	Mat imgMask = iniLSF(mask);
	imgMask = abs(imgMask);

	double nu = 0.0001 * 255 * 255;
	int mu = 1;
	int num = 5;
	int epison = 1;
	float step = 0.1;

	Mat LSF = iniLSF.clone();
	for (int index = 0; index < num; ++index)
	{
		LSF = kernelCV(LSF, src, nu, mu, epison, step);  //正常情况检测出来的应该是大于0，但是出现了一个不正常情况，检测出来的是小于0
	}
	Mat LSFBin;
	LSF.convertTo(LSFBin, CV_8UC1);
	cv::findContours(LSFBin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	if (contours[0].size() > (src.rows + src.cols) * 2 * 0.9)  //针对不正常情况
	{
		LSF = -LSF;
		LSF.convertTo(LSFBin, CV_8UC1);
		contours.clear();
		cv::findContours(LSFBin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	}
}