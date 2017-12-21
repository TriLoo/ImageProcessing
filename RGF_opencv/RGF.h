//
// Created by smher on 17-12-18.
//

#ifndef RGF_SALIENCY_RGF_H
#define RGF_SALIENCY_RGF_H

#include "header.h"
#include "JBF.h"

// template supporting int8 & float3d input
template <typename T>
class RGF : public JBF<T>
{
public:
    //using RGF::level_;

    RGF() = default;
    RGF(int rad, double deltaS, double deltaR, int level);
    ~RGF();

    void rollingguidancefilter(std::vector<cv::Mat> &imgOut, cv::Mat &imgIn);
private:
    //int rad, level;
    //double deltaS, deltaR;
};

template <typename T>
RGF<T>::RGF(int rad, double deltaS, double deltaR, int level) : JBF<T>(rad, deltaS, deltaR, level)
{
}

template <typename T>
RGF<T>::~RGF()
{
}

template <typename T>
void RGF<T>::rollingguidancefilter(std::vector<cv::Mat> &imgOut, cv::Mat &imgIn)
{
    std::cout << "level = " << this->level_ << std::endl;

    cv::Mat C, tempMat;
    if(imgIn.channels() == 1)
    {
        C = cv::Mat::ones(imgIn.size(), CV_32F);
        tempMat = cv::Mat_<float>(imgIn.size());
    }
    else
    {
        C = cv::Mat::ones(imgIn.size(), CV_32FC3);
        tempMat = cv::Mat_<cv::Vec3f>(imgIn.size());
    }

    //for(int i = 0; i < this->level_ + 1; ++i)
    for(int i = 0; i < this->level_; ++i)
    {
        if(i == 0)
            this->jointBilateralFilter(tempMat, C, imgIn);             // must use 'this->' before this function
            //this->jointBilateralFilter(tempMat, C, imgIn);             // must use 'this->' before this function
        else
            this->jointBilateralFilter(tempMat, imgOut[i-1], imgIn);     // Or, add 'using ...' at the start of RFG class
        imgOut.push_back(tempMat);                                              // Or, error "No arguments that depend on a template parameter'
        //std::cout << "Add 1 more..." << std::endl;
    }

}

#endif //RGF_SALIENCY_RGF_H

