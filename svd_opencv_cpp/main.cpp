#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	if(argc != 2)
		cout << "no image data input" << endl;

	Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	if(img.data == NULL)
		cout << "no image data load" << endl;

	Mat img_cvt; 
	img.convertTo(img_cvt, CV_32F, 1.0/255);

	int row = img_cvt.rows;
	int col = img_cvt.cols;
	cout << row << endl;
	cout << col << endl;
	cout << "--------------" << endl;
	Mat W(row, col, CV_32F);
	Mat U(row, col, CV_32F);
	Mat VT(row, col, CV_32F);

	SVD::compute(img_cvt, W, U, VT);

	cvNamedWindow("test", CV_WINDOW_AUTOSIZE);
	imshow("test", img_cvt);

	cout << VT.rows << endl;
	cout << VT.cols << endl;
	cout << "--------------" << endl;
	for (int i = 0; i < W.rows; ++i)
	{
		for(int j = 0; j < W.cols; ++j)
			cout << float(W.data[i+j]) ;
		cout << endl;
	}

	return 0;
}
