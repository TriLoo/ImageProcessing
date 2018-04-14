//
// Created by smher on 18-4-4.
//
#include "LKFlow.h"

using namespace std;
using namespace cv;

#define ATF at<float>

//LK::LK(int l) : level_(l)
LK::LK()
{
}

LK::~LK()
{
}

// using ORB to get the keypoints, to be tracked
void LK::clacKps(const cv::Mat &imgA, const cv::Mat &imgB)
{
    Ptr<ORB> detector = ORB::create(50);       // use default setting, except the nfeatures = 50
    detector->detect(imgA, kpsA_, cv::Mat());
    //detector->detect(imgB, kpsB_, cv::Mat());

}

// calculating the optical flow using a 3 * 3 window: [u, v]^T = (A^TA)^{-1}Ab
Point2f LK::calcUV(const cv::Mat &winLeft, const cv::Mat &winRight)
{
}


void LK::calcOF(cv::Mat &imgOut, const cv::Mat &imgInA, const cv::Mat &imgInB)
{
}

/*
Mat LK::calcGaussSubsample(const cv::Mat &imgIn, int l)
{
    int row = imgIn.rows;
    int col = imgIn.cols;

    int rowL = row >> 1;
    int colL = col >> 1;

    Mat Out = Mat::zeros(cv::Size(colL, rowL), CV_32FC1);

    assert(l < level_);
    const int winWidth = winSize_[l];
}

void LK::calcPyramid(const cv::Mat &imgA, const cv::Mat &imgB)
{
    for (int i = 0; i < level_; ++i)
    {
    }
}
*/

#undef ATF
