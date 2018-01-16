//
// Created by smher on 18-1-11.
//

#ifndef WEIGHTEDMAP_WEIGHTEDMAP_H
#define WEIGHTEDMAP_WEIGHTEDMAP_H

#include "headers.h"

struct GuiParams
{
    int r1_, r2_;
    double eps1_, eps2_;
};

class WeightedMap
{
public:
    WeightedMap() = default;
    WeightedMap(int r, int c);
    ~WeightedMap();

    //void weightedmap(cv::Mat& wmBase, cv::Mat& wmDetail, std::vector<cv::Mat>& imgIns);
    void weightedmap(std::vector<cv::Mat> &wmBase, std::vector<cv::Mat> &wmDetail, std::vector<cv::Mat> &imgIns);
    void setParams(double c = 0.95, int ar = 20, int gr = 3, int guiR = 30, double gs = 0.3, double ge = 0.0001)
    {
        c_ = c;
        GuiRad_ = guiR;     // radius of guided filter
        AvgRad_ = ar;       // radius of average filter
        GauRad_ = gr;       // radius of gaussian filter
        GauSig_ = gs;       // sigma  of gaussian filter
        GuiEps_ = ge;       // eps in guided filter
    }
private:
    void guidedfilter(cv::Mat& imgOut, const cv::Mat& imgInI, const cv::Mat& imgInP, int rad, double eps);
    void localsaliency(cv::Mat& sal, const cv::Mat& imgIn);
    void globalsaliency(cv::Mat& imgOut, const cv::Mat& imgIn);
    void saliencydetection(cv::Mat& sal, const cv::Mat& imgIn);
    //void hcsingle(cv::Mat& imgOut, const cv::Mat& imgIn);
    //std::shared_ptr<WeightedMapImpl> pImpl;

    double c_;    // the factor between local & global saliency map
    int GuiRad_, AvgRad_, GauRad_;
    double GauSig_, GuiEps_;
    int row_, col_;
};

#endif //WEIGHTEDMAP_WEIGHTEDMAP_H
