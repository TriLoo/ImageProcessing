//
// Created by smher on 18-3-5.
//

#ifndef PROJECT_GUIDEDFILTER_H_H
#define PROJECT_GUIDEDFILTER_H_H

#include "headers.h"

// Guided filter, color image only
class GFilter
{
public:
    GFilter() = default;
    GFilter(int r, int c);
    GFilter(GFilter& a) = delete;
    GFilter& operator=(const GFilter& a) = delete;
    ~GFilter();

    void setParams(int r = 45, double e = 10^(-6))
    {
        rad_ = r;
        eps_ = e;
    }
    void setRowCol(int r = 512, int c = 512)
    {
        row_ = r;
        col_ = c;
    }
    void guidedfilter(cv::Mat& imgOut, const cv::Mat& imgInI, const cv::Mat& imgInP);
    void guidedfilterOpenCV(cv::Mat& imgOut, const cv::Mat& imgInI, const cv::Mat& imgInP);
    void boxfilterNpp(cv::Mat& imgOut, const cv::Mat&imgIn, int rad = 16);
    void boxfilterTest(cv::Mat& imgOut, const cv::Mat& imgIn, int rad = 16);

    void initTexture(float* data);
    void initTexture(const cv::Mat& inI, const cv::Mat& inP);
private:
    void guidedfilterSingle(cv::Mat& imgOut, const cv::Mat& imgInI);
    void guidedfilterDouble(cv::Mat& imgOut, const cv::Mat& imgInI, const cv::Mat& imgInP);
    // all parameters are stored in device memory
    //void boxfilter(float* imgOut_d, const float* imgIn_d, int rad);
    void boxfilterTempC(float4* imgIO);
    void boxfilterImgI(float4* imgIO);
    void boxfilterMultiple(float4* corrI, float4* corrIp);
    void gaussianfilter(float* imgOut_d, const float* imgIn_d, int rad, double sig);
    void restoreFromFloat4(float* out, float* in);

    int row_, col_;   // unit: pixel, NOT Byte
    int rad_ = 0;
    double eps_ = 0.0f;

    // used for time testing
    cudaEvent_t startEvent_, stopEvent_;
    float elapsedTime_ = 0.0;
    cudaError_t cudaState_ = cudaSuccess;

    size_t pitch_;
    float4 *tempA_, *tempB_, *tempC_, *tempD_;
    float4 *tempE_, *tempF_;
    float4 *tempData_;   // used in boxfilter

    // used for changed input images
    cv::Mat inI_, inP_;

    // cuda Stream declaration
    cudaStream_t stream1_, stream2_;
};

#endif //PROJECT_GUIDEDFILTER_H_H

