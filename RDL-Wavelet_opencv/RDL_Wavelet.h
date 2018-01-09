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
    virtual ~RDLWavelet() = default;

    void RdlWavelet(std::vector<cv::Mat>& imgOuts, const cv::Mat& imgIn);
private:
    void Horizontal_Predict(cv::Mat& imgOut, const cv::Mat& imgIn);
    void Horizontal_Update(cv::Mat& imgOut, const cv::Mat& imgIn);
};

void RDLWavelet(std::vector<cv::Mat> &imgOuts, const cv::Mat& imgIn);
void SincInterpolation(cv::Mat& imgOut, cv::Mat& imgIn);

#define INDX(r, c, w) ((r) * (w) + (c))

#endif //RDL_WAVELET_RDL_WAVELET_H
