//
// Created by smher on 18-1-8.
//

#ifndef RDL_WAVELET_RDL_WAVELET_H
#define RDL_WAVELET_RDL_WAVELET_H

// All images are stored in 'float' type

#include "headers.h"

class RDLWavelet
{
public:
    RDLWavelet() = default;
    // 只能赋予一次默认实参，通常在声明中完成
    RDLWavelet(int r, int c, int d = 4);      // the number of default direction is 4
    //RDLWavelet(int r, int c, int d);      // the number of default direction is 4
    virtual ~RDLWavelet() = default;

    void RdlWavelet(std::vector<cv::Mat>& imgOuts, const cv::Mat& imgIn);
    void inverseRdlWavelet(cv::Mat &imgOut, std::vector<cv::Mat>& imgIns);
private:
    void Horizontal_Predict(cv::Mat& imgPre, const cv::Mat& imgIn);
    void Horizontal_Update(cv::Mat& layerBase, const cv::Mat& layerDetail, const cv::Mat& imgIn);
    void Inverse_Horizontal_Update(cv::Mat& imgOut,const cv::Mat& imgBase, const cv::Mat& imgDetail);

    cv::Mat Horizontal_SincInterpolation(const cv::Mat& imgIn);

    int row, col, Dir;
};

//void RDLWavelet(std::vector<cv::Mat> &imgOuts, const cv::Mat& imgIn);
//void SincInterpolation(cv::Mat& imgOut, cv::Mat& imgIn);

#define INDX(r, c, w) ((r) * (w) + (c))

#endif //RDL_WAVELET_RDL_WAVELET_H
