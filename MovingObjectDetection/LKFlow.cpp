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
void LK::calcKps(const cv::Mat &imgA)
{
    Ptr<ORB> detector = ORB::create(100);       // use default setting, except the nfeatures = 50
    detector->detect(imgA, kpsA_, cv::Mat());
    //detector->detect(imgB, kpsB_, cv::Mat());

    cout << "size of kpsA = " << kpsA_.size() << endl;
}

// calculating the optical flow using a 3 * 3 window: [u, v]^T = (A^TA)^{-1}Ab
// TODO: Use Eigen3
Point2f LK::calcUV(const cv::Mat &winU, const cv::Mat& winV, const cv::Mat &winb)
{
    //assert(winU.rows == 9 && winV.rows == 9 && winb.rows == 9);

    /*
    Mat matA = Mat::zeros(Size(2, 9), CV_32FC1);

    for (int i = 0; i < 9; ++i)
    {
        matA.ATF(i, 0) = winU.ATF(i, 0);
        matA.ATF(i, 1) = winV.ATF(i, 0);
    }
    */

    Eigen::Matrix2f ATA;
    Eigen::Vector2f Ab;

    float sumA = 0.0, sumB = 0.0, sumC = 0.0, sumD = 0.0, sumE = 0.0;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
        {
            sumA += pow(winU.ATF(i, j), 2.0);
            sumB += winU.ATF(i, j) * winV.ATF(i, j);
            sumD += pow(winV.ATF(i, j), 2.0);

            sumC += winU.ATF(i, j) * winb.ATF(i, j);
            sumE += winV.ATF(i, j) * winb.ATF(i, j);
        }
    }

    ATA(0, 0) = sumA;      // use () to access the eigen elements
    ATA(0, 1) = sumB;
    ATA(1, 0) = sumB;
    ATA(1, 1) = sumD;

    Ab(0) = -1 * sumC;
    Ab(1) = -1 * sumE;

    //Eigen::Matrix2f tempA = ATA.inverse();
    Eigen::Vector2f tempA = ATA.inverse() * Ab;

    return cv::Point2f(tempA(1), tempA(0));
}

void LK::calcGradientUV(cv::Mat& imgOutA, cv::Mat& imgOutB, const cv::Mat &imgIn)
{
    //Mat_<float> kernelVertic, kernelHori;
    Mat kernelVertic, kernelHori;
    kernelVertic = (Mat_<float>(3, 3) << -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
    kernelHori   = (Mat_<float>(3, 3) << -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);

    kernelHori /= 8;
    kernelVertic /= 8;

    filter2D(imgIn, imgOutA, imgIn.depth(), kernelHori);
    filter2D(imgIn, imgOutB, imgIn.depth(), kernelVertic);
}

void LK::calcGradientT(cv::Mat& imgOut, const cv::Mat& imgInA, const cv::Mat &imgInB)
{
    imgOut = imgInB - imgInA;
}


void LK::calcOF(cv::Mat &imgOut, cv::Mat &imgInA, const cv::Mat &imgInB)
{
    calcKps(imgInA);

    Mat gradU, gradV, gradT;
    Mat inA = imgInA;
    Mat inB = imgInB;
    Mat Out = imgInB;

    imgInA.convertTo(inA, CV_32FC3, 1.0);
    imgInB.convertTo(inB, CV_32FC3, 1.0);

    //if (inA.channels() != 1)
        //cvtColor(imgInA, inA, CV_BGR2GRAY);
    if (inA.depth() != CV_32FC1)
        inA.convertTo(inA, CV_32FC1, 1.0);
    //if (inB.channels() != 1)
        //cvtColor(imgInB, inB, CV_BGR2GRAY);
    if (inB.depth() != CV_32FC1)
        inB.convertTo(inB, CV_32FC1, 1.0);

    calcGradientUV(gradU, gradV, imgInA);
    calcGradientT(gradT, imgInA, imgInB);

    const int row = imgInA.rows;
    const int col = imgInA.cols;
    cout << "row = " << row << " col = " << col << endl;

    Mat tempU, tempV, tempT;

    //cout << kpsA_[0].pt << endl;
    //cout << "x = " << kpsA_[0].pt.x << endl;

    //cout << "Step 1 Success." << endl;
    for (const auto& ele : kpsA_)
    {
        Point2f leftPoint = ele.pt;                         // returen (col, row);   ! ! !
        if ((leftPoint.x < 2) || (leftPoint.x > col-2))     // col
        {
            cout << "Out range = " << leftPoint << endl;
            continue;
        }
        if (leftPoint.y < 2 || leftPoint.y > row-2)      // row
        {
            continue;
        }

        tempU = gradU(Range(leftPoint.y-1, leftPoint.y + 1), Range(leftPoint.x - 1, leftPoint.x + 1));
        tempV = gradV(Range(leftPoint.y-1, leftPoint.y + 1), Range(leftPoint.x - 1, leftPoint.x + 1));
        tempT = gradT(Range(leftPoint.y-1, leftPoint.y + 1), Range(leftPoint.x - 1, leftPoint.x + 1));

        //tempU = tempU.reshape(1, 9);    // 9 * 1
        //tempV = tempV.reshape(1, 9);    // 9 * 1
        //tempT = tempT.reshape(1, 9);    // 9 * 1

        Point2f rightPoint = calcUV(tempU, tempV, tempT);

        cout << "[u, v] = " << rightPoint << endl;
        kpsB_.push_back(KeyPoint(rightPoint, 31));
    }

    cout << "Size of Kps B = " << kpsB_.size() << endl;

    for (int i = 0; i < kpsB_.size(); ++i)
    {
        Point2f newDim = kpsB_[i].pt + kpsA_[i].pt;
        cout << "new Dim = " << newDim << endl;
        circle(imgInA, kpsA_[i].pt, 3, Scalar(0, 0, 255));
        circle(Out, newDim, 3, Scalar(0, 0, 255));  // (0, 0, 255): RED
    }

    Out.convertTo(imgOut, CV_8UC3, 1.0);
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
