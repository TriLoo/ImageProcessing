//#include <iostream>

#include "GOL.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    //Mat imgIn = imread("Barbara.jpg", IMREAD_COLOR);
    Mat imgIn = imread("lena.jpg", IMREAD_COLOR);

    if(imgIn.empty())
    {
        cerr << "Read Image failed ..." << endl;
        return -1;
    }

    Mat imgOut = Mat::zeros(imgIn.size(), CV_32FC3);

    imshow("Input", imgIn);
    imgIn.convertTo(imgIn, CV_32FC3, 1.0 / 255);


    GOL<float> gl(1, Size(5, 5), 5);
    gl.gaussoflap(imgOut, imgIn);

    imgOut.convertTo(imgOut, CV_8U, 255);

    imshow("GOL Output", imgOut);


    waitKey(0);

    return 0;
}

