#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat predict(const Mat &_src, const Mat &_filter)
{
	int row = _src.rows;
	int col = _src.cols;

	Mat dst =  Mat::zeros(row, col, CV_32F);

	float temp = 0;
	
	for(int i=0; i < row; ++i)
	{
		for(int j=0;j<col; ++j)
		{
			if(i ==0 || j == 0)
				dst.at<float>(i, j) = _src.at<float>(i, j);
			else if(i == row-1 || j == col-1)
				dst.at<float>(i, j) = _src.at<float>(i, j);
			else
			{ temp = _src.at<float>(i-1, j-1)*_filter.at<float>(0,0) + _src.at<float>(i-1, j+1, 0)*_filter.at<float>(0,1);
				temp += _src.at<float>(i+1, j-1)*_filter.at<float>(1,0) + _src.at<float>(i+1, j+1, 0)*_filter.at<float>(1,1);
				dst.at<float>(i, j) = _src.at<float>(i, j) - temp  ;
			}

		}
	}

	return dst;
}

Mat update(const Mat &_src, const Mat &_filter)
{
	int row = _src.rows;
	int col = _src.cols;

	Mat dst =  Mat::zeros(row, col, CV_32F);

	float temp = 0;
	
	for(int i=0; i < row; ++i)
	{
		for(int j=0;j<col; ++j)
		{
			if(i ==0 || j == 0)
				dst.at<float>(i, j) = _src.at<float>(i, j);
			else if(i == row-1 || j == col-1)
				dst.at<float>(i, j) = _src.at<float>(i, j);
			else
			{ temp = _src.at<float>(i-1, j-1)*_filter.at<float>(0,0) + _src.at<float>(i-1, j+1, 0)*_filter.at<float>(0,1);
				temp += _src.at<float>(i+1, j-1)*_filter.at<float>(1,0) + _src.at<float>(i+1, j+1, 0)*_filter.at<float>(1,1);
				dst.at<float>(i, j) = _src.at<float>(i, j) + temp  ;
			}

		}
	}

	return dst;
}

Mat irlnsw(const Mat &_src,  const Mat &_predict)
{
	int row = _src.rows;
	int col = _src.cols;

	float temp = 0;

	Mat dst = Mat::zeros(row, col, CV_32F);

	for(int i=0; i<row; ++i)
	{
		for(int j=0;j<col; ++j)
		{
			if(i==0 || j == 0)
				dst.at<float>(i, j) = _src.at<float>(i, j);
			else if(i == row || j == col)
				dst.at<float>(i, j) = _src.at<float>(i, j);
			else
			{
				temp = _src.at<float>(i-1, j-1)*_predict.at<float>(0, 0) + _src.at<float>(i-1, j+1)*_predict.at<float>(0, 1); 
				temp += _src.at<float>(i+1, j-1)*_predict.at<float>(1, 0) + _src.at<float>(i+1, j+1)*_predict.at<float>(1, 1); 
				dst.at<float>(i, j) = _src.at<float>(i, j) + temp;
			}
		}
	}

	return dst;
}

int main(int argc, char **argv)
{
	if(argc != 2)
		cout << "no image data input ..." << endl;
	Mat M = imread(argv[1], IMREAD_GRAYSCALE);
	Mat img = Mat::zeros(M.rows, M.cols, CV_32F);

	M.convertTo(img, CV_32F, 1.0/255);

	Mat dst = Mat::zeros(M.rows, M.cols, CV_32F);

	//prepare the filters
	Mat predictM = Mat::zeros(2, 2, CV_32F);
	Mat updateM = Mat::zeros(2, 2, CV_32F);
	predictM.at<float>(0, 0) = 1;
	predictM.at<float>(0, 1) = 0.75;
	predictM.at<float>(1, 0) = 0.75;
	predictM.at<float>(1, 1) = 0.5625;

	updateM.at<float>(0, 0) = 1.0/2;
	updateM.at<float>(0, 1) = 0.75/2;
	updateM.at<float>(1, 0) = 0.75/2;
	updateM.at<float>(1, 1) = 0.5625/2;

	Mat high(dst);
	Mat low(dst);

	high = predict(img, predictM);
	low = update(high, updateM);
	//low = rlnsw(img, update);

	double maxi, mini;
	
	//invert rlnsw calculate
	Mat inv = irlnsw(img, predictM);
	//minMaxLoc(inv, &mini, &maxi);
	/*
	cout << "-----------inv------------" << endl;
	cout << maxi << endl;
	cout << mini << endl;
	for(int i=1; i<high.rows; ++i)
	{
		for(int j=1; j<high.cols; ++j)
		{
			inv.at<float>(i, j) -= mini;
		}
	}
	*/
	minMaxLoc(inv, &mini, &maxi);
	inv.convertTo(dst, CV_8UC1, 255.0/maxi);
	imwrite("dest.jpg", dst);

	minMaxLoc(high, &mini, &maxi);
	for(int i=1; i<high.rows; ++i)
	{
		for(int j=1; j<high.cols; ++j)
		{
			high.at<float>(i, j) -= mini;
		}
	}

	Mat res(high);
	minMaxLoc(high, &mini, &maxi);

	double scale = 255.0/maxi;
	high.convertTo(res, CV_8UC1, scale);

	imwrite("test.jpg", res);

	return 0;
}

