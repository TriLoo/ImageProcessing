#include <iostream>
#include <time.h>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat matSum(Mat &A)
{
	//Mat B(A);
	Mat B = Mat::ones(1, A.cols, CV_32F);
	float sum = 0;
	for(int j = 0; j < A.cols; ++j)  // sum the col direction
	{
		sum = 0;
		for(int i = 0; i < A.rows; ++i)
		{
			sum += (float)A.at<float>(i, j);
		}
		B.at<float>(0, j) = sum;
	}

	return B;
}

int main(int argc, char **argv)
{
	if(argc != 4)
	{
		cout << "Usage : nmf image rank maxiter" << endl;
		return -1;
	}

	cout << "Copyright(C)smher" << endl;

	string imgname = argv[1];
	int rank = stoi(argv[2]);
	int maxiter = stoi(argv[3]);

	Mat M = imread(imgname, CV_LOAD_IMAGE_GRAYSCALE);
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", M);
	int row = M.rows;
	int col = M.cols;
	M.convertTo(M, CV_32F, 1.0);
	Mat W(row, rank, CV_32F);
	Mat H(rank, col, CV_32F);
	Mat One = Mat::ones(row, 1, CV_32F);

	//result mat
	Mat res(M);

	cout << "1" << endl;

	double start = clock();

	randu(W, Scalar::all(0), Scalar::all(1));
	W = W / (One * matSum(W));
	randu(H, Scalar::all(0), Scalar::all(1));

	cout << "2" << endl;

	for(int i = 0; i < maxiter; ++i)
	{
		H = H.mul(W.t() * (M/(W*H)));
		W = W.mul(M*(H.t())/(W*H*H.t()));
		W = W/(One * matSum(W));
	}
	double end = clock();

	cout << "time = " << endl;
	cout << " " << (end - start ) / CLOCKS_PER_SEC << endl;

	res = W * H;
	res.convertTo(res, CV_8UC1, 1);
	namedWindow("restored", WINDOW_AUTOSIZE);
	imshow("restored", res);

	waitKey(0);

	return 0;
}

