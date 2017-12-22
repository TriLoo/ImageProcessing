//
// Created by smher on 17-12-22.
//

#ifndef SALIENCYDETECTION_GLOBALSALIENCY_H
#define SALIENCYDETECTION_GLOBALSALIENCY_H

#include "headers.h"

template <typename T>
class globalSaliency
{
public:
    globalSaliency() = default;
    globalSaliency(int row, int col);

    ~globalSaliency();

    void globalsaliency(cv::Mat &imgOut, const cv::Mat &imgIn);
private:
    void hcsingle(cv::Mat &imgOut, const cv::Mat &imgIn);
    void hccolor(cv::Mat &imgOut, const cv::Mat &imgIn);

};

template <typename T>
globalSaliency<T>::globalSaliency(int row, int col)
{
}

template <typename T>
globalSaliency<T>::~globalSaliency()
{
}

template <typename T>
void globalSaliency<T>::globalsaliency(cv::Mat &imgOut, const cv::Mat &imgIn)
{
    if(imgIn.channels() == 1)
        hcsingle(imgOut, imgIn);
    else
        hccolor(imgOut, imgIn);
}

template <typename T>
void globalSaliency<T>::hcsingle(cv::Mat &imgOut, const cv::Mat &imgIn)
{
    if(imgIn.depth() == CV_32F)
        imgIn.convertTo(imgIn, CV_8U, 255);

    int dims = 256;
    int histSize[] = {256};
    float ranges[] = {0, 256};
    const float *histRange[] = {ranges};

    float Kfactor = imgIn.rows * imgIn.cols;

    bool uniform = true, accumulate = false;

    //cv::Mat hist = cv::Mat_<int>(dims, 1, 0);
    cv::Mat hist;
    cv::calcHist(&imgIn, 1, 0, cv::Mat(), hist, 1, histSize, histRange, uniform, accumulate);

    hist /= Kfactor;

    //std::cout << hist << std::endl;
    //std::vector<float> dist(0);
    cv::Mat lut = cv::Mat_<float>(cv::Size(1, 256), 0);

    for(int i = 0; i < dims; ++i)
    {
        float tempSum = 0.0;
        for (int j = 0; j < dims; ++j)
            tempSum += hist.at<float>(j) * fabs(j - i);
        //dist.push_back(tempSum);
        lut.at<float>(i) = tempSum;
    }

    cv::LUT(imgIn, lut, imgOut);

    cv::normalize(imgOut, imgOut, 0, 1, CV_MINMAX);

    /*   // for test, can work
    std::cout << "hist type = " << hist.type() << ", CV_32S" << CV_32S << std::endl;

    std::cout << hist << std::endl;
    long int tempSum = 0;
    for(auto beg = hist.begin<float>(); beg != hist.end<float>(); ++beg)
    {
        tempSum += *beg;
        std::cout << "*beg = " << *beg << std::endl;
    }

    long int ref = imgIn.rows * imgIn.cols;
    std::cout << "Sum = " << tempSum << std::endl;
    std::cout << "Ref = " << ref << std::endl;
    assert(tempSum == ref);
    */
}

// TODO: I don't know how to get the histgram of color image
template <typename T>
void globalSaliency<T>::hccolor(cv::Mat &imgOut, const cv::Mat &imgIn)
{
    cv::Mat imgInLab;
    cv::cvtColor(imgIn, imgInLab, CV_RGB2Lab);
}

#endif //SALIENCYDETECTION_GLOBALSALIENCY_H
