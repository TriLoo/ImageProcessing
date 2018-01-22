//
// Author : smh - 2017.01.16
//

#include "GuidedFilter.h"

using namespace std;
using namespace cv;

void imgShow(Mat img)
{
    imshow("Temp", img);
    waitKey(0);
}

GFilter::GFilter(int r, int c) : row_(r), col_(c), rad_(45), eps_(0.000001)
{
}

GFilter::~GFilter()
{
}

// Kernel functions
// do boxfilter
__global__ void
d_boxfilter_rgb_x(float* d_out, const float3 * __restrict__ d_in, int row, int col)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.x + blockDim.x * blockIdx.x;
}

void GFilter::boxfilter(float *imgOut_d, const float *imgIn_d, int rad)
{
}

void GFilter::boxfilterTest(cv::Mat &imgOut, const cv::Mat &imgIn, int rad)
{
    const float* imgInP = (float *)imgIn.data;
}

void GFilter::gaussianfilter(float *imgOut_d, const float *imgIn_d, int rad, double sig)
{
}

// 输入图像是相同的  e.g. imgInI == imgInP
// color image guided filter
void GFilter::guidedfilterSingle(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{

}

// 输入图像是不同的  e.g. imgInI != imgInP
void GFilter::guidedfilterDouble(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{

}

void GFilter::guidedfilter(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    assert(imgInP.channels() == 3 && imgInI.channels() == 3);
    //const float *imgA = (float *)imgInI.data;
    //const float *imgB = (float *)imgInP.data;
    equal_to<const float*> T;
    if (T((float *)imgInI.data, (float*)imgInP.data))
        guidedfilterSingle(imgOut, imgInI, imgInI);
    else
        guidedfilterDouble(imgOut, imgInI, imgInP);
}

// Contrast Experiments
void GFilter::guidedfilterOpenCV(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    assert(imgInP.channels() == 3);
    if (rad_ == 0)
        setParams(16, 0.01);    // Image Enhancement

    Mat meanI, corrI, varI, meanP;
    boxFilter(imgInI, meanI, imgInI.depth(), Size(rad_, rad_));
    boxFilter(imgInI.mul(imgInI), corrI, imgInI.depth(), Size(rad_, rad_));
    boxFilter(imgInP, meanP, imgInP.depth(), Size(rad_, rad_));
    varI = corrI - meanI.mul(meanI);
    //imgShow(varI);

    vector<Mat> vecP(imgInP.channels()), vecI(imgInI.channels());
    vector<Mat> vecMeanI(imgInI.channels()), vecMeanP(imgInP.channels());
    split(imgInP, vecP);
    split(imgInI, vecI);
    split(meanP, vecMeanP);
    split(meanI, vecMeanI);

    Mat covIp, sameP, sameMeanP, meanA, meanB;
    vector<Mat> vecA(imgInI.channels());
#pragma unloop
    for (int i = 0; i < 3; ++i)
    {
        //vector<Mat> vecSameP{vecP[i], vecP[i], vecP[i]};
        //merge(vecSameP, sameP);
        //boxFilter(imgInI.mul(sameP), covIp, imgInI.depth(), Size(rad_, rad_));
        //vector<Mat> vecSameMeanP{vecMeanP[i], vecMeanP[i], vecMeanP[i]};
        //merge(vecSameMeanP, sameMeanP);
        cvtColor(vecP[i], sameP, CV_GRAY2BGR);         // use cvtColor to do the broadcast purpose, instead of above method
        cvtColor(vecMeanP[i], sameMeanP, CV_GRAY2BGR);
        boxFilter(imgInI.mul(sameP), covIp, imgInI.depth(), Size(rad_, rad_));
        covIp = covIp - meanI.mul(sameMeanP);

        Mat a = covIp / (varI + eps_);
        boxFilter(a, meanA, a.depth(), Size(rad_, rad_));
        //cout << "a.channels = " << a.channels() << endl;         // for test

        split(a, vecA);
        Mat b = vecMeanP[i] - (vecA[0].mul(vecMeanI[0]) + vecA[1].mul(vecMeanI[1]) + vecA[2].mul(vecMeanI[2]));
        boxFilter(b, meanB, b.depth(), Size(rad_, rad_));
        //cout << "b.channels = " << b.channels() << endl;         // for test

        split(meanA, vecA);
        vecP[i] = vecA[0].mul(vecI[0]) + vecA[1].mul(vecI[1]) + vecA[2].mul(vecI[2]) + meanB;
    }
    merge(vecP, imgOut);
}
