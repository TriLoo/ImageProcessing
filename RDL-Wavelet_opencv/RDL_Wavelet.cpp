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

    imgOuts.push_back(SincImg);
}

void SincInterpolation(cv::Mat& imgOut, cv::Mat& imgIn)
{

}
