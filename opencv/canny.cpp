//确定边缘点之后进行延伸
void trace(cv::Mat &gradImg, cv::Mat &edge, const int r, const int c, const int low)
{
	int w = gradImg.cols;
	int h = gradImg.rows;
	if (r < 0 || c < 0 || r >= h || c >= w)
		return;
	if (edge.at<uchar>(r, c) == 0)  //如果不是0，就代表已经确定为边缘点
	{
		edge.at<uchar>(r, c) = 255;
		for (int i = -1; i <= 1; ++i)  //3*3的矩阵进行搜索
		{
			for (int j = -1; j <= 1; ++j)
			{
				if ((r + i) >= h || (c + j) >= w || (r + i) < 0 || (c + j) < 0)
					continue;
				float val = gradImg.at<float>(r + i, c + j);
				if(val < low)
					continue;
				trace(gradImg, edge, r + i, c + j, low);			
			}
		}
	}
}

//此Canny算法与opencv的有一定差异，感觉opencv的canny边缘更薄
//此处未进行滤波，可以尝试用双边滤波代替高斯滤波（双边滤波可以保留边缘信息）
void myCanny(const cv::Mat src, cv::Mat &dst, const int low, const int high)
{
	cv::Mat img = src.clone();
	cv::Mat dx, dy; 
	cv::Mat edge = cv::Mat::zeros(img.size(), CV_8UC1); 
	Mat gradientImg = cv::Mat::zeros(img.size(), CV_32FC1);  //梯度幅值
	cv::Sobel(img, dx, CV_16SC1, 1, 0);
	cv::Sobel(img, dy, CV_16SC1, 0, 1);

	bool L2gradient = false;
	if (L2gradient)
	{
		Mat _dx, _dy;
		dx.convertTo(_dx, CV_32FC1);
		dy.convertTo(_dy, CV_32FC1);
		cv::magnitude(_dx, _dy, gradientImg);   //magnitude-->sqrt(x*x, y*y)
	}		
	else
	{
		Mat _dx, _dy;
		dx.convertTo(_dx, CV_32FC1);
		dy.convertTo(_dy, CV_32FC1);
		gradientImg = abs(_dx) + abs(_dy);
	}
		

	cv::Mat gradientNMSImg = cv::Mat::zeros(gradientImg.size(), CV_32FC1);
	for (int r = 1; r < gradientImg.rows - 1; ++r)
	{
		for (int c = 1; c < gradientImg.cols - 1; ++c)
		{
			float _x = dx.at<short>(r, c);
			float _y = dy.at<short>(r, c);
			float angle = std::atan2f(_y, _x) / CV_PI * 180;  //当前梯度方向（与边缘方向正交）
			float gradVal = gradientImg.at<float>(r, c); //当前梯度的幅值
			//根据梯度方向得到边缘方向并进行量化--0，45，90，135
			//atan2 --> (-180, 180]		划重点：是沿着梯度方向对幅值进行非极大值抑制，而非边缘方向，这里初学者容易弄混。
			//垂直边缘--梯度方向为0或180（偏差±22.5）
			//NMS
			//垂直边缘--梯度方向为水平方向-3*3邻域内左右方向比较
			if (abs(angle) < 22.5 || abs(angle) > 157.5) 
			{
				float left = gradientImg.at<float>(r, c - 1);
				float right = gradientImg.at<float>(r, c + 1);
				if (gradVal >= left && gradVal >= right)
					gradientNMSImg.at<float>(r, c) = gradVal;
			}
			//水平边缘--梯度方向为垂直方向-3*3邻域内上下方向比较
			else if ((angle >= 67.5 && angle <= 112.5) || (angle >= -112.5 && angle <= -67.5)) 
			{
				float top = gradientImg.at<float>(r - 1, c);
				float down = gradientImg.at<float>(r + 1, c);
				if (gradVal >= top && gradVal >= down)
					gradientNMSImg.at<float>(r, c) = gradVal;
			}
			//+45°边缘--梯度方向为其正交方向-3*3邻域内右上左下方向比较
			else if ((angle > 112.5 && angle <= 157.5) || (angle > -67.5 && angle <= -22.5)) 
			{
				float right_top = gradientImg.at<float>(r - 1, c + 1);
				float left_down = gradientImg.at<float>(r + 1, c - 1);
				if (gradVal >= right_top && gradVal >= left_down)
					gradientNMSImg.at<float>(r, c) = gradVal;
			}
			//+135°边缘--梯度方向为其正交方向-3*3邻域内右下左上方向比较
			else if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5)) 
			{
				float left_top = gradientImg.at<float>(r - 1, c - 1);
				float right_down = gradientImg.at<float>(r + 1, c + 1);
				if (gradVal >= left_top && gradVal >= right_down)
					gradientNMSImg.at<float>(r, c) = gradVal;
			}
		}
	}
	//利用高低阈值进行连接
	for (int i = 1; i < img.rows - 1; ++i)
	{
		for (int j = 1; j < img.cols - 1; ++j)
		{
			float val = gradientNMSImg.at<float>(i, j);  
			if (val >= high)
				trace(gradientNMSImg, edge, i, j, low);
			else if (val < low)
				gradientNMSImg.at<float>(i, j) = 0;
		}
	}
	dst = edge.clone();
	std::cout << "myCanny" << std::endl;
}