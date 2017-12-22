//
// Created by smher on 17-12-22.
//

#ifndef SALIENCYDETECTION_FTSALIENCY_H
#define SALIENCYDETECTION_FTSALIENCY_H

#include "headers.h"

template <typename T>
class FTSaliency
{
public:
    FTSaliency() = default;
    ~FTSaliency();

    void ftsaliency(cv::Mat &imgOut, const cv::Mat &imgIn);
private:

};

template <typename T>
FTSaliency<T>::~FTSaliency()
{
}

template <typename T>
void FTSaliency<T>::ftsaliency(cv::Mat &imgOut, const cv::Mat &imgIn)
{
    cv::Mat imgInLab;
    cv::cvtColor(imgIn, imgInLab, CV_RGB2Lab);

    int chans = imgIn.channels();
    std::vector<cv::Mat> LabImg(chans);
    //std::cout << "Depth of imgInLab = " << imgInLab.depth() << std::endl;

    cv::Scalar saMean = cv::mean(imgInLab);

    // For test
    std::cout << saMean << std::endl;
    //cv::Mat meanImg = cv::Mat(imgIn.size(), imgIn.depth(), saMean);

    // For test
    //std::cout << meanImg.depth() << std::endl;

    cv::Mat gaussImg, tempImg;
    cv::GaussianBlur(imgInLab, gaussImg, cv::Size(5, 5), 1, 1, cv::BORDER_REPLICATE);
    // For test
    //std::cout << gaussImg.depth() << std::endl;

    LOG("Begin Calculate imgOut.");
    //imgOut = meanImg - gaussImg;
    tempImg = gaussImg - saMean;
    //meanImg -= gaussImg;
    LOG("After Calculate imgOut.");

    tempImg.mul(tempImg);
    // For test
    //std::cout << tempImg.channels() << std::endl;
    cv::split(tempImg, LabImg);
    for (auto& ele : LabImg)
        imgOut += ele;
    LOG("Accumulate all Mats in LabImg.");
    //cv::cvtColor(imgOut, imgOut, CV_Lab2RGB);
    //cv::cvtColor(imgOut, imgOut, CV_RGB2GRAY);

    cv::normalize(imgOut, imgOut, 0, 1, CV_MINMAX);
    LOG("Normalized the imgOut to 0 and 1.");
}

#endif //SALIENCYDETECTION_FTSALIENCY_H
