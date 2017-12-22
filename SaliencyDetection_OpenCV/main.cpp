//#include <iostream>
#include "globalSaliency.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    //Mat imgIn = imread("Barbara.jpg", IMREAD_COLOR);
    Mat imgIn = imread("lena.jpg", IMREAD_COLOR);

    try
    {
        if(imgIn.empty())
            throw runtime_error("Read Image failed ...");
    }
    catch (runtime_error err)
    {
        cerr << err.what() << endl;
        return -1;
    }

    imshow("Input", imgIn);

    cvtColor(imgIn, imgIn, COLOR_RGB2GRAY);

    globalSaliency<uint8_t> gs;

    Mat imgOut = Mat::zeros(imgIn.size(), CV_32F);
    gs.globalsaliency(imgOut, imgIn);

    imgOut.convertTo(imgOut, CV_8UC1, 255.0);

    imshow("HC Output", imgOut);


    waitKey(0);

    return 0;
}