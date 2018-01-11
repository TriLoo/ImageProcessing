//
// Created by smher on 18-1-11.
//

#ifndef WEIGHTEDMAP_GUIDEDFILTER_H
#define WEIGHTEDMAP_GUIDEDFILTER_H

#include "headers.h"

class GuidedFilter
{
public:
    //GuidedFilter() = default;
    GuidedFilter(int r = 10, double e = EPS);
    virtual ~GuidedFilter();

    void guidedfilter(cv::Mat& imgOut, const cv::Mat& imgInI, cv::Mat& imgInP);
private:
    int rad_;
    double eps_;
};

#endif //WEIGHTEDMAP_GUIDEDFILTER_H
