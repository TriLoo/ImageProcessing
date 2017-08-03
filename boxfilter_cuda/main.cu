//#include <iostream>
#include "boxfilter.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

#define RAD 3
#define ROW 10
#define COL 10
#define SIZE (ROW * COL)

int main(int argc, char **argv)
{
	cout << "hello world ..." << endl;

	float *img = new float [SIZE];
	for(int i = 0; i < 100; i++)
		img[i] = 1;

	BFilter bf;
	bf.rad = 3;
	bf.width = 10;
	bf.height = 10;
	bf.data = img;

	cout << "Input data : " << endl;
	bf.print();

	bf.boxfilter();

	cout << "Result : " << endl;
	bf.print();

	return 0;
}
