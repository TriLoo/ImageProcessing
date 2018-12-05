//
// Created by smher on 18-1-8.
//

#include "RDL_Wavelet.h"

#define DIRECTION 4

using namespace std;
using namespace cv;

// construction function
RDLWavelet::RDLWavelet(int r, int c, int d) : row(r), col(c), Dir(d)
{
}

// Sinc函数的计算过程等价于三个一维横向的滤波，产生三个滤波结果，然后加上原图共四个原图大小的矩阵, 然后把三个矩阵的对应位置的值交替拼接在一起，就实现了这个函数的功能
// 三个一维横向滤波用到的Kernel函数就是Sinc矩阵的三行。
Mat RDLWavelet::Horizontal_SincInterpolation(const cv::Mat &imgIn)
{
    int mySample = 4;

    int inRows = imgIn.rows;
    int inCols = imgIn.cols;
    Mat imgRet(inRows, inCols*4, CV_32FC1);

    Mat Sinc = (Mat_<float>(3, 8) << -0.0110, 0.0452, -0.1437, 0.8950, 0.2777, -0.0812, 0.0233, -0.0158,
                                     -0.0105, 0.0465, -0.1525, 0.6165, 0.6165, -0.1525, 0.0465, -0.0105,
                                     -0.0053, 0.0233, -0.0812, 0.2777, 0.8950, -0.1437, 0.0452, -0.0110);
    //cout << "Shape of Sinc Mat: " << Sinc.rows << " " << Sinc.cols << endl;
    //cout << Sinc << endl;

    float sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    for(int i = 0; i < inRows; ++i)
    {
        for(int j = 0; j < inCols; ++j)
        {
            imgRet.at<float>(i, j * 4) = imgIn.at<float>(i, j);  // 输入原始图像的值位于输出图像4个数中的最左边位置上
            for (int l = -mySample + 1; l <= mySample; ++l)
            {
                int x = j + l;
                if(x < 0)
                    x = -x + 2;
                if(x >= inCols)
                    x = j * 2 - x;

                sum1 += imgIn.at<float>(i, x) * Sinc.at<float>(0, l + mySample - 1);
                sum2 += imgIn.at<float>(i, x) * Sinc.at<float>(1, l + mySample - 1);
                sum3 += imgIn.at<float>(i, x) * Sinc.at<float>(2, l + mySample - 1);
            }
            imgRet.at<float>(i, j * 4 + 1) = sum1;
            imgRet.at<float>(i, j * 4 + 2) = sum2;
            imgRet.at<float>(i, j * 4 + 3) = sum3;

            sum1 = 0;
            sum2 = 0;
            sum3 = 0;
        }
    }

    return imgRet;
}

void RDLWavelet::Horizontal_Predict(cv::Mat &imgPre, const cv::Mat &imgIn)
{
    int row = imgIn.rows;
    int col = imgIn.cols;

    Mat SincImg = Mat::zeros(Size(4 * col, row), CV_32F);

    // Interpolation basing on Sinc
    //resize(imgIn, SincImg, Size(4 * col, row), 4, 0, INTER_CUBIC);
    SincImg = Horizontal_SincInterpolation(imgIn);

    const int Dir = DIRECTION;
    const float Divd = 2 * ( 2 * Dir + 1 );

    // Pad border
    Mat SincImg_buf(Size(SincImg.cols + 2 * Dir, SincImg.rows + 2 * Dir), CV_32F);
    //copyMakeBorder(SincImg, SincImg_buf, Dir, Dir, Dir, Dir, BORDER_REFLECT101);
    copyMakeBorder(SincImg, SincImg_buf, Dir, Dir, Dir, Dir, BORDER_WRAP);

    // begin horizontal predict
    Mat tempMat = Mat::zeros(imgIn.size(), CV_32F);
    float tempSum = 0.0;
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
    imgPre = tempMat;
}

void RDLWavelet::Horizontal_Update(cv::Mat &layerBase,const cv::Mat &layerDetail, const cv::Mat &imgIn)
{
    int row = imgIn.rows;
    int col = imgIn.cols;
    const int Dir = DIRECTION;
    const float Divd = ((Dir << 1) + 1) << 1;
    // begin Horizontal Update
    Mat SincImgUpdate, SincImgUpdate_buf;
    // for test
    Mat imgTest(imgIn.size(), CV_32F);
    SincImgUpdate = Horizontal_SincInterpolation(layerDetail);
    copyMakeBorder(SincImgUpdate, SincImgUpdate_buf, Dir, Dir, Dir, Dir, BORDER_WRAP);
    float tempSum = 0.0;
    //for (int i = Dir; i < Dir + row - 1; ++i)
    for (int i = Dir; i < Dir + row; ++i)
    {
        auto rowPtrUp = SincImgUpdate_buf.ptr<float>(i - 1);
        auto rowPtrDown = SincImgUpdate_buf.ptr<float>(i + 1);
        for (int j = 0; j < col; ++j)
        {
            for (int k = -Dir; k <= Dir; ++k)
                tempSum += rowPtrUp[Dir * j + k + Dir] + rowPtrDown[Dir * j + k + Dir];
            imgTest.at<float>(i - Dir, j) = imgIn.at<float>(i - Dir, j) + (tempSum / Divd);
            tempSum = 0.0;
        }
    }
    layerBase =imgTest;
}

void RDLWavelet::Inverse_Horizontal_Update(cv::Mat &imgOut, const cv::Mat& imgBase, const cv::Mat& imgDetail)
{
    const int colT = imgBase.cols;
    const int rowT = imgBase.rows;
    //const float Dvid = 1.0 / (((Dir << 1)  + 1) << 1);
    const float Dvid = 1.0 / (2 * (Dir * 2 + 1));
    Mat tempMat, tempMat_buf;
    Mat resMat(Size(colT, rowT), imgBase.type());
    tempMat = Horizontal_SincInterpolation(imgDetail);
    copyMakeBorder(tempMat, tempMat_buf, Dir, Dir, Dir, Dir, BORDER_WRAP);

    float tempSum = 0.0;
    for (int i = Dir; i < Dir + rowT; ++i)
    {
        auto rowPtrUp = tempMat_buf.ptr<float>(i - 1);
        auto rowPtrDown = tempMat_buf.ptr<float>(i + 1);
        for (int j = 0; j < colT; ++j)
        {
            for (int k = -Dir; k <= Dir; ++k)
                tempSum += rowPtrUp[j*Dir + Dir + k] + rowPtrDown[j * Dir + Dir + k];
            resMat.at<float>(i - Dir, j) = imgBase.at<float>(i - Dir, j) - (tempSum * Dvid);
            tempSum = 0.0;
        }
    }
    imgOut = resMat;
}

void RDLWavelet::RdlWavelet(std::vector<cv::Mat> &imgOuts, const cv::Mat &imgIn)
{
    const int row = imgIn.rows;
    const int col = imgIn.cols;
    const int Dir = DIRECTION;
    //Mat tempMat(Size(col * Dir, row), imgIn.type());

    Mat imgH_Detail(imgIn.size(), imgIn.type());
    Mat imgH_Base(imgIn.size(), imgIn.type());
    Horizontal_Predict(imgH_Detail, imgIn);   // 他妈的是对的啊
    Horizontal_Update(imgH_Base, imgH_Detail, imgIn);

    // Vertical Predict & Update
    // Part I : LL, LH
    Mat imgV_L = imgH_Base.t();           // work as input image
    Mat imgV_H = imgH_Detail.t();         // work as input image

    Mat imgV_Detail(imgIn.size(), imgIn.type());    // i.e. LH
    Mat imgV_Base(imgIn.size(), imgIn.type());      // i.e. LL

    // begin predict
    Horizontal_Predict(imgV_Detail, imgV_L);   // LH
    // begin update
    Horizontal_Update(imgV_Base, imgV_Detail, imgV_L);   // LL

    imgOuts.push_back(imgV_Base.t());        // LL
    imgOuts.push_back(imgV_Detail.t());      // LH

    // Part II : HL, HH
    Horizontal_Predict(imgV_Detail, imgV_H);
    Horizontal_Update(imgV_Base, imgV_Detail, imgV_H);

    imgOuts.push_back(imgV_Base.t());                // i.e. HL
    imgOuts.push_back(imgV_Detail.t());              // i.e. HH
}

void RDLWavelet::inverseRdlWavelet(cv::Mat &imgOut, std::vector<cv::Mat> &imgIns)
{
    assert(imgIns.size() == 4);    // Only includes LL, LH, HL, HH, four elements, i.e. A, H, V, D
    Mat imgBase(Size(col, row), imgIns[0].type());
    Mat imgDetail(Size(col, row), imgIns[0].type());
    Mat tempMat(Size(col, row), imgIns[0].type());

    // Inverse Vertical Transform
    Inverse_Horizontal_Update(imgDetail, imgIns[2].t(), imgIns[3].t());
    Inverse_Horizontal_Update(imgBase, imgIns[0].t(), imgIns[1].t());

    // Inverse Horizontal Transform
    Inverse_Horizontal_Update(imgOut, imgBase.t(), imgDetail.t());
}

