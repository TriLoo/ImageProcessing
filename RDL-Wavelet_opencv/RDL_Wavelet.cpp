//
// Created by smher on 18-1-8.
//

#include "RDL_Wavelet.h"

#define DIRECTION 4

using namespace std;
using namespace cv;

void imgShow(const Mat& img)
{
    imshow("Temp", img);
    waitKey(0);
}

void imgShow(const Mat&& img)
{
    imshow("Temp", img);
    waitKey(0);
}

RDLWavelet::RDLWavelet(int r, int c, int d) : row(r), col(c), Dir(d)
{
}

void RDLWavelet::Horizontal_Predict(cv::Mat &imgPre, const cv::Mat &imgIn)
{
    int row = imgIn.rows;
    int col = imgIn.cols;

    Mat SincImg = Mat::zeros(Size(4 * col, row), CV_32F);

    // Interpolation basing on Sinc
    resize(imgIn, SincImg, Size(4 * col, row), 4, 0, INTER_CUBIC);

    const int Dir = DIRECTION;
    const float Divd = 2 * ( 2 * Dir + 1 );

    // Pad border
    Mat SincImg_buf(Size(SincImg.cols + 2 * Dir, SincImg.rows + 2 * Dir), CV_32F);
    copyMakeBorder(SincImg, SincImg_buf, Dir, Dir, Dir, Dir, BORDER_REFLECT101);

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
    resize(layerDetail, SincImgUpdate, Size(4*col, row), 4, 0, INTER_CUBIC);
    copyMakeBorder(SincImgUpdate, SincImgUpdate_buf, Dir, Dir, Dir, Dir, BORDER_REFLECT101);
    float tempSum = 0.0;
    for (int i = Dir; i < Dir + row - 1; ++i)
    {
        auto rowPtrUp = SincImgUpdate.ptr<float>(i - 1);
        auto rowPtrDown = SincImgUpdate.ptr<float>(i + 1);
        for (int j = 0; j < col; ++j)
        {
            for (int k = -Dir; k < Dir; ++k)
                tempSum += rowPtrUp[Dir * j + k + Dir] + rowPtrDown[Dir * j + k + Dir];
            imgTest.at<float>(i - Dir, j) = imgIn.at<float>(i - Dir, j) + (tempSum / (Divd * 2));
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
    resize(imgDetail, tempMat, Size(Dir * colT, rowT), Dir, 0, INTER_CUBIC);
    copyMakeBorder(tempMat, tempMat_buf, Dir, Dir, Dir, Dir, BORDER_REFLECT101);

    float tempSum = 0.0;
    for (int i = Dir; i < Dir + rowT - 1; ++i)
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
    Horizontal_Predict(imgH_Detail, imgIn);
    Horizontal_Update(imgH_Base, imgH_Detail, imgIn);
    // for test
    // imgOuts.push_back(imgH_Base);
    // imgOuts.push_back(imgH_Detail);

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

    imgOuts.push_back(imgV_Base.t());
    imgOuts.push_back(imgV_Detail.t());

    // Part II : HL, HH
    Horizontal_Predict(imgV_Detail, imgV_H);
    Horizontal_Update(imgV_Base, imgV_Detail, imgV_H);

    imgOuts.push_back(imgV_Base.t());                // i.e. HL
    imgOuts.push_back(imgV_Detail.t());              // i.e. HH
}

void RDLWavelet::inverseRdlWavelet(cv::Mat &imgOut, std::vector<cv::Mat> &imgIns)
{
    assert(imgIns.size() == 4);    // Only includes LL, LH, HL, HH, four elements
    Mat imgBase(Size(col, row), imgIns[0].type());
    Mat imgDetail(Size(col, row), imgIns[0].type());
    Mat tempMat(Size(col, row), imgIns[0].type());

    // Inverse Vertical Transform
    Inverse_Horizontal_Update(imgDetail, imgIns[2].t(), imgIns[3].t());
    Inverse_Horizontal_Update(imgBase, imgIns[0].t(), imgIns[1].t());
    //imgShow(imgBase);

    // Inverse Horizontal Transform
    Inverse_Horizontal_Update(imgOut, imgBase.t(), imgDetail.t());
}

/*
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
    imgOuts.push_back(SincImg);   // for test
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
            cout << SincImg.at<float>(i,j) << ", ";
        cout << endl;
    }
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
*/
