//
// Created by smher on 18-1-8.
//

#include "RDL_Wavelet.h"

using namespace std;
using namespace cv;

void RDLWavelet(std::vector<cv::Mat> &imgOuts, const cv::Mat& imgIn)
{
    int row = imgIn.rows;
    int col = imgIn.cols;
    // 对图像进行插值
    // In Size() constructor, the number of rows and the number of columns go in the inverse order
    Mat SincImg = Mat::zeros(Size(4 * col, row), CV_32F);

    // Interpolation basing on Sinc
    resize(imgIn, SincImg, Size(4 * col, row), 4, 0, INTER_CUBIC);
    // for test
    /*
    imgOuts.push_back(SincImg);   // for test
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
            cout << SincImg.at<float>(i,j) << ", ";
        cout << endl;
    }
    */
    const int Dir = 4;
    const float Divd = 2 * ( 2 * Dir + 1 );

    // Pad border
    Mat SincImg_buf(Size(SincImg.cols + 2 * Dir, SincImg.rows + 2 * Dir), CV_32F);
    copyMakeBorder(SincImg, SincImg_buf, Dir, Dir, Dir, Dir, BORDER_REFLECT101);

    // begin horizontal predict
    Mat tempMat = Mat::zeros(imgIn.size(), CV_32F);
    float tempSum = 0.0;
    cout << "Begin Horizontal Predict Success." << endl;
    for (int i = Dir; i < Dir + row; ++i)
    {
        auto rowPtrUp = SincImg_buf.ptr<float>(i - 1);
        auto rowPtrDown = SincImg_buf.ptr<float>(i + 1);
        for (int j = 0; j < col; ++j)
        {
            for (int k = -Dir; k <= Dir; ++k)
                tempSum += rowPtrUp[Dir * j + k + Dir] + rowPtrDown[Dir * j + Dir + k];
            tempMat.at<float>(i-Dir, j) = imgIn.at<float>(i-Dir, j) - (tempSum / Divd);
            tempSum = 0.0;
        }
    }
    cout << "Horizontal Predict Success." << endl;
    imgOuts.push_back(tempMat);

    // begin Horizontal Update
    Mat SincImgUpdate, SincImgUpdate_buf;
    // for test
    Mat imgTest(tempMat.size(), CV_32F);
    resize(tempMat, SincImgUpdate, Size(4*col, row), 4, 0, INTER_CUBIC);
    copyMakeBorder(SincImgUpdate, SincImgUpdate_buf, Dir, Dir, Dir, Dir, BORDER_REFLECT101);
    for (int i = Dir; i < Dir + row - 1; ++i)
    {
        auto rowPtrUp = SincImg_buf.ptr<float>(i - 1);
        auto rowPtrDown = SincImg_buf.ptr<float>(i + 1);
        for (int j = 0; j < col; ++j)
        {
            for (int k = -Dir; k < Dir; ++k)
                tempSum += rowPtrUp[Dir * j + k + Dir] + rowPtrDown[Dir * j + k + Dir];
            imgTest.at<float>(i - Dir, j) = imgIn.at<float>(i - Dir, j) + (tempSum / (Divd * 2));
            //tempMat.at<float>(i - Dir, j) = imgIn.at<float>(i - Dir, j) + (tempSum / (Divd * 2));   // for test, same as former
            tempSum = 0.0;
        }
    }
    imgOuts.push_back(imgTest);
}

void SincInterpolation(cv::Mat& imgOut, cv::Mat& imgIn)
{

}
