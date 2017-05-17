#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

/*
void Sobel(Mat &inImg, Mat &outImg)
{
	int row = 
}
*/

// matrix initialization
void InitMat(Mat *Res, const vector<float>& A)
{
	int row = Res->rows;
	int col = Res->cols;

	auto beg = A.begin();
	auto end = A.end();

	for(int i = 0; i < row; i++)
		for(int j = 0; j < col; j++)
		{
			Res->at<float>(i,j) = *beg;
			beg++;
			if(beg == end)
				break;
		}
}

void Sobel(Mat &imgIn, Mat &imgOut, Mat &sX, Mat &sY)
{
	int imgRow = imgIn.rows;
	int imgCol = imgIn.cols;
	int sRow = sX.rows;
	int sCol = sX.cols;

	//Mat imgX = Mat::zeros(imgRow, imgCol);
	//Mat imgY = Mat::zeros(imgRow, imgCol);
	float X = 0; 
	float Y = 0;
	float T = 0;  // Temp value

	float lu = 0;
	float lm = 0; 
	float lb = 0; 
	float mu = 0; 
	float mm = 0; 
	float mb = 0;
	float ru = 0; 
	float rm = 0; 
	float rb = 0;

	for(int i = 1; i < imgRow - 1; i++)
	{
		for(int j = 1; j < imgCol - 1; j++)
		{
			lu = imgIn.at<float>(i - 1, j - 1);
			lm = imgIn.at<float>(i, j - 1);
			lb = imgIn.at<float>(i + 1, j - 1);
			mu = imgIn.at<float>(i - 1, j);
			mm = imgIn.at<float>(i, j);
			mb = imgIn.at<float>(i + 1, j);
			ru = imgIn.at<float>(i - 1, j + 1);
			rm = imgIn.at<float>(i, j + 1);
			rb = imgIn.at<float>(i + 1, j + 1);

			// calculat the Sobel
			//imgX.at<float>(i, j) = lu*sX.at<float>(0,0) + mu*sX.at<float>(0,1) + ru*sX.at<float>(0,2) + lm*sX.at<float>(1,0) + mm*sX.at<float>(1,1) + rm*sX.at<float>(1,2) + lb*sX.at<float>(2,0) + mb*sX.ar<float>(2,1) + rb*sX.at<float>(2,2);
			X = lu*sX.at<float>(0,0) + mu*sX.at<float>(0,1) + ru*sX.at<float>(0,2) + lm*sX.at<float>(1,0) + mm*sX.at<float>(1,1) + rm*sX.at<float>(1,2) + lb*sX.at<float>(2,0) + mb*sX.at<float>(2,1) + rb*sX.at<float>(2,2);

			Y = lu*sY.at<float>(0,0) + mu*sY.at<float>(0,1) + ru*sY.at<float>(0,2) + lm*sY.at<float>(1,0) + mm*sY.at<float>(1,1) + rm*sY.at<float>(1,2) + lb*sY.at<float>(2,0) + mb*sY.at<float>(2,1) + rb*sY.at<float>(2,2);

			//imgOut.at<float>(i, j) = sqrt(X * X + Y * Y);
			imgOut.at<float>(i,j) = sqrt(X * X + Y * Y);

			//if(T > )
		}
	}

	//imgOut = 
}

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		cout << "no image name input ..." << endl;
		return -1;
	}
	Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	if(!img.data)
	{
		cout << "read image data failed ..." << endl;
		return -1;
	}

	//const int 
	Mat sobelX = Mat::zeros(3, 3, CV_32F);
	Mat sobelY = Mat::zeros(3, 3, CV_32F);

	vector<float> Ax = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	vector<float> Ay = {-1, -1, -1, 0, 0, 0, 1, 2, 1};

	InitMat(&sobelX, Ax);
	InitMat(&sobelY, Ay);
	
	img.convertTo(img, CV_32F, 1.0);
	
	int row = img.rows;
	int col = img.cols;
	Mat res = Mat::zeros(row, col, CV_32F);

	Sobel(img, res, sobelX, sobelY);

	res.convertTo(res, CV_8UC1, 1.0);

	imwrite("restore.jpg", res);

	return 0;
}
