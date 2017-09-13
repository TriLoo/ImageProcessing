#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "boxfilter.h"
#include <ctime>

using namespace cv;

#define FILTERW 11

int main(int argc, char **argv)
{
	cout << "hello world ..." << endl;

	Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);

	if(!img.data){
		cerr << "Read image failed ..." << endl;
		return -1;
	}

	img.convertTo(img, CV_32F, 1.0);

	clock_t start, stop;

	int wid = img.cols;
	int hei = img.rows;

	float *filter = new float [FILTERW * FILTERW];

	for(int i = 0; i < FILTERW * FILTERW; ++i)
	{
		filter[i] = 1.0 / (FILTERW*FILTERW);
	}

	cout << "filter = " << filter[26] << endl;

	Mat imgOut = Mat::zeros(hei, wid, CV_32F);

	float *imgIn, *imgOutP;
	imgIn = (float *)img.data;
	imgOutP = (float *)imgOut.data;

	BFilter bf(wid, hei, FILTERW);

	start = clock();
	//bf.boxfilterTex(imgOutP, imgIn, wid, hei, filter, FILTERW);      // slowest
	bf.boxfilterSep(imgOutP, imgIn, wid, hei, FILTERW);                // Fastest
	//bf.boxfilterGlo(imgOutP, imgIn, wid, hei, filter, FILTERW);      // second slowest
	//bf.boxfilterSha(imgOutP, imgIn, wid, hei, filter, FILTERW);      // second fast
	stop = clock();
	double dur = (double)(stop - start) / CLOCKS_PER_SEC * 1000;

	cout << "Boxfilter Based on Global Memory: " << dur << " ms" << endl;

	imgOut.convertTo(imgOut, CV_8UC1, 1.0);

	imshow("Output", imgOut);

	waitKey(0);

	delete [] filter;

	return 0;
}
