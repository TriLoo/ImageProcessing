#include <iostream>
#include <time.h>
#include <string>
#include <opencv2/highgui.hpp> 
#include <opencv2/imgproc.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

#define EPS 2.2204E-5

// sum the column direction of A, and return a row vector
Mat matSum(Mat &A)
{
	Mat B = Mat::zeros(1, A.cols, CV_32F);
	float sum = 0;
	for(int j = 0; j < A.cols; ++j)   // sum the col direction
	{
		for(int i = 0; i < A.rows; ++i)
		{
			sum += (float)A.at<float>(i, j);
		}
		B.at<float>(0, j) = sum;
	}

	return B;
}

// sum all the element of A, and return a float value
float matSumF(Mat &A) 
{
	float sum = 0;
	Mat B = matSum(A);
	for(int i = 0; i < B.cols; ++i)
		sum += B.at<float>(0, i);

	return sum;
}

// calculate the result rank satisfied some rule
int rankChoose(Mat &A)
{
	int rank = 0;
	float sum = matSumF(A);   // sum of Mat A
	sum = sum * 0.9;
	float tempsum = 0;

	// get the elements by 'at()'
	for(int i = 0; i < A.rows; ++i)
	{
		tempsum += A.at<float>(i, 0);
		if(tempsum > sum)
		{
			rank = i;
			break;
		}
	}

	return rank;
}

// calculate the absulute value of Mat
void matAbs(Mat &A)
{
	float temp = 0;
	for(int i = 0; i < A.rows; ++i)
		for(int j = 0; j < A.cols; ++j)
		{
			temp = A.at<float>(i, j);
			if(temp < 0)
				A.at<float>(i, j) = (-1) * temp;
		}
}

// reshape the mat from vector into diag matrix
void matDiag(Mat &A, int col)
{
	int row = A.rows;    // based on SVD, A.rows == M.rows!!!
	Mat B = Mat::zeros(row, col, CV_32F);
	for(int i = 0; i < row; ++i)
	{
		B.at<float>(i, i) = A.at<float>(i, 0);
	}

	A = B;
}

/* -------------vim for test-----------------*/
Mat matDiagM(Mat &A)
{
	int row = A.rows;
	Mat B = Mat::zeros(row, row, CV_32F);
	for(int i = 0; i < row; ++i)
		B.at<float>(i, i) = A.at<float>(i, 0);

	return B;
}
/* -------------vim for test-----------------*/

int main(int argc, char **argv)
{
	// test args
	if(argc != 3)
	{
		cout << "Usage : nmf image.name maxiter" << endl;
		return -1;
	}

	cout << "Copyright(C)smher	version2.0" << endl;

	string imgname = argv[1];
	//string imgname = "lena.jpg";
	int maxiter = stoi(argv[2]);
	//int	maxiter = 100;
	cout << "maxiter = " << maxiter << endl;    // for test

	//read image
	Mat M = imread(imgname, CV_LOAD_IMAGE_GRAYSCALE);
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", M);
	int row = M.rows;
	int col = M.cols;

	//convert inputted image from unsigned char into float
	M.convertTo(M, CV_32F, 1.0);

	//define WS\US\VTS used in SVD calculation
	Mat WS(row, col, CV_32F);           // the calculated singular values
	Mat US(row, col, CV_32F);			// the calculated left singular vectors
	Mat VTS(row, col, CV_32F);			// transposed matrix of right singular vectors

	// output signals to user to imply where the software is running at
	cout << "0" << endl;  				// for test


	// calculated the svd of inputted image ...
	SVD::compute(M, WS, US, VTS, SVD::FULL_UV);

	// calculate the result rank
	int rank = rankChoose(WS);
	cout << "rank = " << rank << endl;

	//define W & H matrix
	Mat W = Mat::zeros(row, rank, CV_32F);
	Mat H = Mat::zeros(rank, col, CV_32F);

	// define a column vector
	Mat One = Mat::ones(row, 1, CV_32F);

	// define the result mat
	Mat res = Mat::zeros(row, col, CV_32F);

	cout << "2" << endl;    			// for test

	// initialize the W & H matrix
	W = US(Range::all(), Range(0, rank));
	matDiag(WS, M.cols);
	H = WS(Range(0, rank), Range::all()) * VTS;
	cout << "2.2" << endl;

	// get the absolute value of W & H
	matAbs(W);
	matAbs(H);

	cout << "3" << endl;
	double start = clock();

	Mat H_curr(H);
	Mat W_curr(W);
	
	W_curr = W;
	H_curr = H;

	Mat H_next = Mat::ones(H.rows, H.cols, CV_32F);

	H.copyTo(H_next);

	int a = 0; 
	cout << &a << endl;
	cout << &H.datastart << endl;
	cout << &H_curr.datastart << endl;

	H_curr += EPS;
	H_next += EPS;
	W_curr += EPS;
	//W_next = W_next + EPS;

	// multiplicative calculate
	for(int i = 0; i < maxiter; ++i)
	{
		H_next = H_next.mul(W_curr.t() * (M/(W_curr*H_next)));
		W_curr = W_curr.mul(M*(H_curr.t() + EPS)/(W_curr*H_curr*H_curr.t() + EPS));
		//W_curr = W_curr/(One * matSum(W_curr));
		H_next.copyTo(H_curr);
	}

	double end = clock();
	cout << "time = " << (end-start)/CLOCKS_PER_SEC << endl;
	cout << "4" << endl;

	W = W_curr;
	H = H_curr;

	// restore the image using obtained W & H
	res = W * H;

	res.convertTo(res, CV_8UC1, 1);

	// show the resotred image
	namedWindow("restored", WINDOW_AUTOSIZE);
	imshow("restored", res);

	// waitKey !!! to hold the image windows
	waitKey(0);

	return 0;
}

