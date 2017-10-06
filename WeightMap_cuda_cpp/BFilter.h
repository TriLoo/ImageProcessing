//
// Created by smher on 17-9-29.
//

#ifndef GUIDEDFILTER_BFILTER_H
#define GUIDEDFILTER_BFILTER_H

#include "headers.h"

class BFilter
{
public:
    BFilter() = default;
    BFilter(int wid, int hei);
    virtual ~BFilter();

    void boxfilter(float *d_imgOut, float *d_imgIn, int wid, int hei, int filterR);
    void boxfilterTest(float *imgOut, float *imgIn, int wid, int hei, int filterR);
private:
    float *d_imgIn_, *d_imgOut_;
};

#endif //GUIDEDFILTER_BFILTER_H
