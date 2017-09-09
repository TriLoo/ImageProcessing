//
// Created by smher on 17-9-1.
//

#ifndef GFFFUSION_BOXFILTER_H
#define GFFFUSION_BOXFILTER_H

#include "headers.h"

class BFilter
{
public:
    BFilter() = default;
    ~BFilter() = default;

    // data is stored on global memory
    void boxfilter(float *d_out, const float *d_in, int wid, int hei, const float * __restrict__ d_filter, int filterW);
    // data is stored on 2D pitch or 2D array
    void boxfilter(float *d_out, const float *d_in, size_t pitch, int wid, int hei, const float * __restrict__ d_filter, int filterW);
private:
};

#endif //GFFFUSION_BOXFILTER_H
