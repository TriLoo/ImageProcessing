#include "calHist.h"
#include <assert.h>
#include <stdexcept>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

int main(int argc, char **argv)
{
	/*
	if(argc < 2)
	{
		cout << "no image read ..." << endl;
		return -1;
	}
	*/
	//Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img = imread("lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	try
	{
		if(img.data == NULL)
			throw runtime_error("read image failed ...");
	}
	catch(runtime_error err)
	{
		cerr << err.what() << endl;
		return -1;
	}

	int row = img.rows;
	int col = img.cols;
	const int SIZE = row * col;

	unsigned char * buffer = (unsigned char *)img.data;

	unsigned int *hist = new unsigned int [256];
	float *histF = new float [256];

	// CPU time
	float start, stop;
	float duration;

	start = clock();

	// call calculate histogram function
	calHist(buffer, SIZE, hist);

	stop = clock();
	duration = double(stop - start) / CLOCKS_PER_SEC * 1000;

	cout << "CPU time : " << duration << " ms" << endl;

	long sumH = 0;
	for(int i = 0; i < 256; i++)
		sumH += hist[i];
	assert(sumH == SIZE);

	for(int i = 0; i < 256; i++)
		histF[i] = 1.0 * hist[i] / SIZE;

	float sumF = 0.0;
	for(int i = 0; i < 256; i++)
		sumF += histF[i];

	cout << "sumF = " << sumF << endl;

	delete [] hist;
	delete [] histF;

	return 0;
}
