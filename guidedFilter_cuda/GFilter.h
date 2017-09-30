//
// Created by smher on 17-9-29.
//

#ifndef GUIDEDFILTER_GFILTER_H
#define GUIDEDFILTER_GFILTER_H

#include "headers.h"
#include "BFilter.h"

//class GFilter : public BFilter
class GFilter
{
public:
    //GFilter() = default;
    GFilter(int wid, int hei);
    ~GFilter();

    void guidedfilter(float *d_imgOut, float *d_imgInI, float *d_imgInP, int wid, int hei, int rad, double eps);
    void guidedfilterTest(float *imgOut, float *imgInI, float *imgInP, int wid, int hei, int rad, double eps);
private:
    //float *d_imgIn_, *d_imgOut_;
    float *d_tempA_, *d_tempB_, *d_tempC_, *d_tempD_;
};

#endif //GUIDEDFILTER_GFILTER_H
