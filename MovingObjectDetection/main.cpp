#include "ViBe.h"

using namespace std;
using namespace cv;

// 帧差法
Mat FrameMinus(const Mat& imgA, const Mat& imgB)
{
    Mat Res;
    Res.create(imgA.size(), CV_32FC1);

    Mat tempA, tempB;
    Mat tempRes;

    cvtColor(imgA, tempA, CV_BGR2GRAY);
    tempA.convertTo(tempA, CV_32FC1, 1.0 / 255);
    cvtColor(imgB, tempB, CV_BGR2GRAY);
    tempB.convertTo(tempB, CV_32FC1, 1.0 / 255);

    //Res = abs(imgA - imgB);
    absdiff(tempA, tempB, tempRes);
    threshold(tempRes, tempRes, 0.2, 1, THRESH_BINARY);
    tempRes.convertTo(tempRes, CV_8UC1, 255);
    //Res = tempA;

    // do Morph structuring element: close operaton
    Mat Kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    // Way 1:
    morphologyEx(tempRes, Res, MORPH_CLOSE, Kernel);
    // Way 2: dilate -> erode
    //dilate(tempRes, tempRes, Kernel);
    //erode(tempRes, Res, Kernel);

    return Res;
}

Mat FrameMinus(const Mat& imgA, const Mat& imgB, Mat& imgIO)
{
    Mat Res;
    Res.create(imgA.size(), CV_32FC1);

    Mat tempA, tempB;
    Mat tempRes;

    cvtColor(imgA, tempA, CV_BGR2GRAY);
    tempA.convertTo(tempA, CV_32FC1, 1.0 / 255);
    cvtColor(imgB, tempB, CV_BGR2GRAY);
    tempB.convertTo(tempB, CV_32FC1, 1.0 / 255);

    //Res = abs(imgA - imgB);
    absdiff(tempA, tempB, tempRes);
    threshold(tempRes, tempRes, 0.2, 1, THRESH_BINARY);
    tempRes.convertTo(tempRes, CV_8UC1, 255);
    //Res = tempA;
    imgIO = tempRes;

    // do Morph structuring element: close operaton
    Mat Kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    // Way 1:
    morphologyEx(tempRes, Res, MORPH_CLOSE, Kernel);
    // Way 2: dilate -> erode
    //dilate(tempRes, tempRes, Kernel);
    //erode(tempRes, Res, Kernel);

    return Res;
}

// 背景差分法: Using mean filter
Mat BackGroundDifference(const vector<Mat> &imgBGs, const Mat& imgCurr)
{
    Mat tempSum = Mat::zeros(imgCurr.size(), CV_32FC1);
    Mat tempImg;

    for (auto & ele : imgBGs)
        tempSum = tempSum + ele;

    size_t N = imgBGs.size();
    Mat imgBG = tempSum / N;

    Mat Res = FrameMinus(imgBG, imgCurr);

    return Res;
}

// TODO: ViBe+

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
    Mat Res, resBefore;
    ViBe tvb;
    tvb.initViBe(imgA.cols, imgA.rows);
    tvb.initialFrame(imgA);
    tvb.detectionBG(imgB);

    Res = tvb.getBGimg();

    //Res = FrameMinus(imgA, imgB, resBefore);

    //imshow("Before Morph", resBefore);
    imshow("Result", Res);

    waitKey(0);

    return 0;
}

