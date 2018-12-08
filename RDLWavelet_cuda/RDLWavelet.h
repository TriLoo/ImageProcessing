//
// Created by smher on 18-12-6.
//

#ifndef RDLWAVELETCUDA_RDLWAVELET_H
#define RDLWAVELETCUDA_RDLWAVELET_H

#include "headers.h"

class RDLWavelet
{
public:
    //RDLWavelet();
    RDLWavelet(int r, int c, int d = 4);   // 函数声明的时候，实现默认值，不要在定义的时候生成
    ~RDLWavelet();

    void setParams(int d);
    void doRDLWavelet(std::vector<cv::Mat> &imgOuts, const cv::Mat& imgIn);
    //void doInverseRDLWavelet(cv::Mat& imgOut, std::vector<cv::Mat> &imgIns);
    void doInverseRDLWavelet(cv::Mat& imgOut);
private:
    // Funcs
    void HorizontalPredict(float *d_imgOut, const float * d_imgIn);
    void HorizontalUpdate(float *d_imgOut, const float * d_imgDetail, const float * d_imgIn);
    void InverseHorizontalUpdate(float *d_imgOut, const float * d_imgBase, const float * d_imgDetail);
    void VerticalPredict(float *d_imgOut, const float * d_imgIn);
    void VerticalUpdate(float *d_imgOut, const float * d_imgDetail, const float * d_imgIn);
    void InverseVerticalUpdate(float *d_imgOut, const float * d_imgBase, const float * d_imgDetail);

    // Datas
    int rows_, cols_;   // shape of input image
    int dir_;   // number of directions
    // memory on CPU, pre-allocated in constructure function
    cv::Mat tempMatA_, tempMatB_;
    float *h_imgIn_, *h_tempOutA_, *h_tempOutB_;
    // memory on GPU, pre-allocated
    // 1024 * 1024 ~ MM, MM,  20M
    // 512 * 512 ~ 1.5M, 1.5M, 6M
    float *d_imgIn_, *d_imgOut_,  *d_Sinc_;
    float *d_tempA_, *d_tempB_, *d_tempC_, *d_tempD_;
    // CUDA Error Check
    cudaError_t cudaStatus_;
    // CUDA Event
    cudaEvent_t startEvent_, stopEvent_;
    // CUDA thread hirearchy
    dim3 blockPerGrid_, threadPerBlock_;
    // CUDA stream
    cudaStream_t  CudaStream_;
};

#endif //RDLWAVELETCUDA_RDLWAVELET_H
