//
// Created by smher on 17-9-15.
//

#ifndef GAUSSFILTER_GAUSSFILTER_H
#define GAUSSFILTER_GAUSSFILTER_H

#include "headers.h"

#define PI 3.131459
//#define FILTERRAD 5
//#define FITLERSIG 5

// gaussian filter
class GFilter
{
public:
    GFilter(int wid, int hei, int filterW, float sig);
    ~GFilter();

    void createfilter();

    void prepareMemory(float *imgIn, int wid, int hei);

    void gaussfilterGlo(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW);
    void gaussfilterTex(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW);
    void gaussfilterCon(float *imgOut, float *imgIn, int wid, int hei, int filterW);
    void gaussfilterSha(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW);
    void gaussfilterSep(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW);
    void gaussfilterShaSep(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW);

private:
    float *d_imgIn_, *d_imgOut_;
    float *d_filter_;
    float *filter_;

    int filterW_, filterSize_, filterR_;
    float sig_;
};

#endif //GAUSSFILTER_GAUSSFILTER_H
