int otsu(const Mat src, Mat &dst)
{
	const int grayScale = 256;
	int grayNum[grayScale] = { 0 };
	int w = src.cols;
	int h = src.rows;
	for (int r = 0; r < h; ++r)	//直方图统计
	{
		const uchar* ptr = src.ptr<uchar>(r);
		for (int c = 0; c < w; ++c)
		{
			grayNum[ptr[c]]++;   
		}
	}

	int pixSum = w * h; //全图的总像素
	double P[grayScale] = { 0 };  //每个灰度级出现的概率
	double PK[grayScale] = { 0 }; //概率累计和 
	double MK[grayScale] = { 0 }; //灰度级的累加均值
	double tmpPK = 0;
	double tmpMK = 0;
	for (int k = 0; k < grayScale; ++k)
	{
		P[k] = 1.0 * grayNum[k] / pixSum;
		PK[k] = tmpPK + P[k];
		tmpPK = PK[k];
		MK[k] = tmpMK + P[k] * k;
		tmpMK = MK[k];
	}

	double maxVariance = 0;
	int indexThresh = 0;
	for (int i = 0; i < grayScale; ++i)
	{
		double _variance = pow((MK[grayScale - 1] * PK[i] - MK[i]), 2) / (PK[i] * (1 - PK[i]));
		if (_variance > maxVariance)
		{
			maxVariance = _variance;
			indexThresh = i;
		}
	}
	return indexThresh;
}