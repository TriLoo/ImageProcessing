#include <iostream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

// 帧差法
Mat FrameMinus(const Mat& imgA, const Mat& imgB)
{
    Mat Res;
    Res.create(imgA.size(), CV_32FC1);

    Mat tempA, tempB;

    cvtColor(imgA, tempA, CV_BGR2GRAY);
    tempA.convertTo(tempA, CV_32FC1, 1.0 / 255);
    cvtColor(imgB, tempB, CV_BGR2GRAY);
    tempB.convertTo(tempB, CV_32FC1, 1.0 / 255);

    //Res = abs(imgA - imgB);
    absdiff(tempA, tempB, Res);
    threshold(Res, Res, 0.2, 1, THRESH_BINARY);
    Res.convertTo(Res, CV_8UC1, 255);
    //Res = tempA;

    return Res;
}

// 光流法



int main(int argc, char **argv) {
    std::cout << "Hello, World!" << std::endl;

    assert(argc == 3);

    string nameA(argv[1]);
    string nameB(argv[2]);

    Mat imgA = imread(nameA, IMREAD_COLOR);
    Mat imgB = imread(nameB, IMREAD_COLOR);

    assert(imgA.empty() == false);
    assert(imgB.empty() == false);

    //Mat Res = Mat::zeros(imgA.size(), imgA.type());
    Mat Res;

    Res = FrameMinus(imgA, imgB);

    imshow("Result", Res);

    waitKey(0);

    return 0;
}

