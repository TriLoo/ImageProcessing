//
// Created by smher on 18-4-4.
//

#include "ViBe.h"

using namespace std;
using namespace cv;

//ViBe::ViBe(int n = 20, int r = 20, int t = 2, int s = 16, int i = 0) : num_samp_(n), R_(r), T_(t), samp_rate_(s), id_(i)
ViBe::ViBe(int n, int r, int t, int s, int i) : num_samp_(n), R_(r), T_(t), samp_rate_(s), id_(i)
{
    neiPos_ = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
}

void ViBe::initViBe(int wid, int hei)
{
    //num_samp_ = n;
    //R_ = r;
    //T_ = t;
    //samp_rate_ = s;
    //id_ = 0;

    bgModels_ = vector<Mat>(num_samp_, Mat::zeros(Size(wid, hei), CV_32FC1));
    imgBG_ = Mat::zeros(Size(wid, hei), CV_32FC1);
    fgTimes_ = Mat::zeros(Size(wid, hei), CV_32FC1);
}

void ViBe::detectionBG(cv::Mat &imgIn)
{
    ++id_;
    Mat tempMat = imgIn;
    RNG rd;
    int row, col;

    if (tempMat.channels() != 1)
        cvtColor(tempMat, tempMat, CV_BGR2GRAY);
    if (tempMat.channels() != 1)
        tempMat.convertTo(tempMat, CV_32FC1, 1.0);

    int dist = 0, sumTemp = 0;
    for (int i = 0; i < imgIn.rows; ++i)
    {
        for (int j = 0; j < imgIn.cols; ++j)
        {
            int pixelVal = tempMat.at<float>(i, j);
            // 计算前景点
            for (int k = 0; k < num_samp_; ++k)
            {
                dist = abs(pixelVal - bgModels_[k].at<float>(i, j));
                if (dist > R_)
                    sumTemp++;
                if (sumTemp >= T_)         // 提前终止, 当前点为前景，设为255
                    break;
            }
            // 更新背景模板
            if (sumTemp >= T_)         // 此时，该点为前景点
            {
                imgBG_.at<float>(i, j) = 255;
                fgTimes_.at<float>(i, j)++;
                // 这部分对应ViBe算法具体更新方法中的第三点
                if (fgTimes_.at<float>(i, j) > 50)   // 连续检测该点为前景超过50次，误将背景点认为前景了
                {
                    int idx = rd.uniform(0, num_samp_);
                    bgModels_[idx].at<float>(i, j) = pixelVal;
                }
            }
            else                       // 此时，该点为背景点
            {
                fgTimes_.at<float>(i, j) = 0;
                imgBG_.at<float>(i, j) = 0;
            }

            // Update Background Model, 这部分对应ViBe算法具体更新方法中的前两点
            if (sumTemp < T_)   // 对应于背景
            {
                int idx = rd.uniform(0, samp_rate_);
                // 更新本像素的背景模型
                if (idx == 0)   // 概率： 1 / samp_rate_
                {
                    int kth = rd.uniform(0, num_samp_);
                    bgModels_[kth].at<float>(i, j) = pixelVal;
                }

                // 更新周围像素
                if (idx == 1)
                {
                    int kth = rd.uniform(0, num_samp_);

                    row = rd.uniform(0, 9);
                    row = i + neiPos_[row];
                    col = rd.uniform(0, 9);
                    col += j + neiPos_[col];

                    if (row < 0)
                        row = 0;
                    if (col < 0)
                        col = 0;

                    if (row > imgIn.rows)
                        row = imgIn.rows - 1;
                    if (col > imgIn.cols)
                        col = imgIn.cols - 1;

                    bgModels_[kth].at<float>(row, col) = pixelVal;
                }

            }
        }
    }
}

// 输入图像是第一帧时，或，背景模型失效时
void ViBe::initialFrame(cv::Mat &imgIn)
{
    ++id_;
    RNG rd;
    int radVal;
    Mat tempMat = imgIn;

    if (tempMat.channels() != 1)
        cvtColor(tempMat, tempMat, CV_BGR2GRAY);
    if (tempMat.channels() != 1)
        tempMat.convertTo(tempMat, CV_32FC1, 1.0);

    int row, col;
    for (int i = 0; i < imgIn.rows; ++i)
    {
        for (int j = 0; j < imgIn.cols; ++j)
        {
            for (int k = 0; k < num_samp_; ++k)
            {
                radVal = rd.uniform(0, 9);
                row = i + neiPos_[radVal];
                radVal = rd.uniform(0, 9);
                col = j + neiPos_[radVal];

                if (row < 0)
                    row = 0;
                if (col < 0)
                    col = 0;

                if (row > imgIn.rows)
                    row = imgIn.rows - 1;
                if (col > imgIn.cols)
                    col = imgIn.cols - 1;

                bgModels_[k].at<float>(i, j) = imgIn.at<float>(row, col);
            }
        }
    }
}


