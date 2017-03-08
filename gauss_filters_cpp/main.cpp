#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main(int argc, char **argv)
{
	if(argc < 2)
		std::cout << "no image data input" << std::endl;

	/*
	IplImage *bar = cvLoadImage(argv[1], 0);
	int a = bar->nChannels;
	std::cout << "gray image " << a << std::endl;
	cvNamedWindow("test", CV_WINDOW_AUTOSIZE);
	cvShowImage("test", bar);

	cvWaitKey(0);
	*/

	IplImage *img = cvLoadImage(argv[1]);
	if(!img->imageData)
		return -1;

	cvNamedWindow("example1", CV_WINDOW_AUTOSIZE);
	cvShowImage("example1", img);

	IplImage *gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	cvConvertImage(img, gray);
	cvNamedWindow("gray", CV_WINDOW_AUTOSIZE);
	cvShowImage("gray", gray);

	int width = gray->width;
	int height = gray->height;

	IplImage *blu = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1);

	int templates[25] = {1, 4, 7, 4, 1,
	4, 16, 26, 16,4,
	7, 26, 41, 26, 7,
	4, 16, 26, 16, 4, 
	1, 4, 7, 4, 1};

	for(int j=2;j<height-2; ++j)
		for(int i=2;i<width-2;++i)
		{
			int sum = 0;
			int index = 0;
			for(int m = j-2; m < j+3; ++m)
			{
				for(int n = i-2; n<i+3; ++n)
				{
					sum += gray->imageData[m*width + n] * templates[index++];
				}
			}
			sum /= 273;
			if(sum > 255)
				sum = 255;

			blu->imageData[j*width + i] = sum;
		}

	cvNamedWindow("blu", CV_WINDOW_AUTOSIZE);
	cvShowImage("blu",blu);

	cvWaitKey(0);
	cvReleaseImage(&img);
	cvDestroyWindow("example1");
	cvReleaseImage(&gray);
	cvDestroyWindow("gray");
	cvReleaseImage(&blu);
	cvDestroyWindow("blu");

	return 0;
}
