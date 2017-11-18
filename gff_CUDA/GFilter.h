//
// Created by smher on 17-11-18.
//

#ifndef GFFFUSIONFINAL_GFILTER_H
#define GFFFUSIONFINAL_GFILTER_H

#include "BFilter.h"

class GFilter : public virtual BFilter
{
public:
    GFilter() = default;
    GFilter(int wid, int hei);
    ~GFilter();

    void guidedfilter(float *d_imgOut, float *d_imgInI, float *d_imgInP, int wid, int hei, int rad, double eps);
    void guidedfilterTest(float *imgOut, float *imgInI, float *imgInP, int wid, int hei, int rad, double eps);
private:
    float *d_tempA_, *d_tempB_, *d_tempC_, *d_tempD_;
};


#endif //GFFFUSIONFINAL_GFILTER_H
