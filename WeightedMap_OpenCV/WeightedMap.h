//
// Created by smher on 18-1-11.
//

#ifndef WEIGHTEDMAP_WEIGHTEDMAP_H
#define WEIGHTEDMAP_WEIGHTEDMAP_H

#include "headers.h"

class WeightedMap
{
public:
    WeightedMap() = default;
    WeightedMap(int r, int c);
    ~WeightedMap();

    void weightedmap(cv::Mat& wmBase, cv::Mat& wmDetail, std::vector<cv::Mat>& imgIns);
    void setParams(int ar, int gr, int guiR, double gs, double ge)
    {
        GuiRad_ = guiR;     // radius of guided filter
        AvgRad_ = ar;       // radius of average filter
        GauRad_ = gr;       // radius of gaussian filter
        GauSig_ = gs;       // sigma  of gaussian filter
        GuiEps_ = ge;       // eps in guided filter
    }
private:
    void guidedfilter(cv::Mat& imgOut, const cv::Mat& imgInI, const cv::Mat& imgInP);
    void localsaliency(cv::Mat& sal, const cv::Mat& imgIn);
    void globalsaliency(cv::Mat& imgOut, const cv::Mat& imgIn);
    void hcsingle(cv::Mat& imgOut, const cv::Mat& imgIn);
    //std::shared_ptr<WeightedMapImpl> pImpl;

    int GuiRad_, AvgRad_, GauRad_;
    double GauSig_, GuiEps_;
    int row_, col_;
};

#endif //WEIGHTEDMAP_WEIGHTEDMAP_H
