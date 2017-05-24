#include "sobel.h"

int main(int argc, char **argv)
{
	/*   for test
	if(argc != 2)
	{
		cerr << "no image data input ..." << endl;
		return -1;
	}
	*/
	//Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img = imread("lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	cout << "Image size is : " << img.rows << " * " << img.cols << endl;

	img.convertTo(img, CV_32F, 1.0);

	const int SIZE = img.rows * img.cols;
	float *dataIn = (float *)img.data;
	float *dataOut = new float [SIZE]();

	mySobel imgSobel(img.rows, img.cols, dataIn);
	imgSobel.SobelCompute(dataIn, dataOut);

	// for test
	/*
	for(int i = 0; i < 10; i++)
	{
		cout << dataOut[i] << endl;
	}
	*/

	//Mat imgRes = Mat::zeros(img.rows, img.cols, CV_32F);
	//imgRes.convertTo(imgRes, CV_32F, 1.0);
	//imgRes.data = dataOut;
	Mat imgRes(img.rows, img.cols, CV_32F, (void *)dataOut, 0);

	imgRes.convertTo(imgRes, CV_8UC1, 1.0);
	//namedWindow("result", CV_WINDOW_AUTOSIZE);
	imshow("result", imgRes);
	imwrite("result.jpg",imgRes);

	waitKey(0);

	delete [] dataOut;
}
