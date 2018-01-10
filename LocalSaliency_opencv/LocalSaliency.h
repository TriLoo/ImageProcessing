//
// Created by smher on 18-1-10.
//

#ifndef LOCALSALIENCY_LOCALSALIENCY_H
#define LOCALSALIENCY_LOCALSALIENCY_H

#include "headers.h"

class LocalSaliency
{
public:
    LocalSaliency() = default;
    LocalSaliency(int r, int c, int ar = 20, int gr = 3, double gs = 0.3);
    ~LocalSaliency() = default;

    void localSaliency(cv::Mat& sal, const cv::Mat& imgIn);

private:

    int row, col;
    int AvgRad, GauRad;
    double GauSig;
};

#endif //LOCALSALIENCY_LOCALSALIENCY_H
