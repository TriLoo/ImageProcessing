#include "twoscale.h"

#define FILTERR 15

// input origin image, and return high pass mat outH, low pass mat outL
void TwoScale(const Mat &inA, Size ksize, Mat &outH, Mat &outL)
{
	// use average filter to get the twoscale decomposing
	blur(inA, outL, ksize);
	outH = inA - outL;
}

int main()
{
	Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
	if(!img.data)
	{
		cerr << "Read image failed ..." << endl;
		return -1;
	}

	clock_t start, stop;
	double duration = 0.0;

	int row = img.rows;
	int col = img.cols;

	imshow("Input", img);

	img.convertTo(img, CV_32F, 1.0);

	float *imgIn = (float *)(img.data);

	Mat imgOutA = Mat::zeros(row, col, CV_32F);
	float *imgOutA_P = (float *)(imgOutA.data);
	Mat imgOutB = Mat::zeros(row, col, CV_32F);
	float *imgOutB_P = (float *)(imgOutB.data);

	cout << 1.0f / ((FILTERR << 1) + 1) << endl;

	TScale ts(col, row);
	start =clock();
	ts.twoscaleTest(imgOutA_P, imgOutB_P, imgIn, col, row, FILTERR);
	stop = clock();

	/*
	BFilter bf(col, row);
	start =clock();
	bf.boxfilterTest(imgOutA_P, imgIn, col, row, FILTERR);
	stop = clock();
	*/

	duration = double((stop - start) * 1.0 / CLOCKS_PER_SEC * 1000.0);
	cout << "Boxfilter On GPU : " << duration << " ms" << endl;

/*
	for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < 10; j++)
			cout << imgOutA_P[j * col + i] << " ,";
		cout << endl;
	}
*/

	// test by OpenCV
	Mat imgOutOpenCV_A = Mat::zeros(row, col, CV_32F);
	Mat imgOutOpenCV_B = Mat::zeros(row, col, CV_32F);
	start = clock();
	TwoScale(img, Size(31, 31), imgOutOpenCV_A, imgOutOpenCV_B);
	stop = clock();
	duration = double((stop - start) * 1.0 / CLOCKS_PER_SEC * 1000.0);
	cout << "Boxfilter On OpenCV : " << duration << " ms" << endl;
	imgOutOpenCV_A.convertTo(imgOutOpenCV_A, CV_8UC1, 1.0);
	imgOutOpenCV_B.convertTo(imgOutOpenCV_B, CV_8UC1, 1.0);
	imshow("High Pass OpenCV", imgOutOpenCV_A);
	imshow("Low Pass OpenCV", imgOutOpenCV_B);


	imgOutA.convertTo(imgOutA, CV_8UC1, 1.0);
	imgOutB.convertTo(imgOutB, CV_8UC1, 1.0);
	imshow("Low Pass GPU", imgOutA);
	imshow("High Pass GPU", imgOutB);


	//waitKey(0);

	return 0;
}
