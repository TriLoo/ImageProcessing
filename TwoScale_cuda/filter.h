//
// Created by smher on 17-9-19.
//

#ifndef TWOSCALE_BOXFILTER_H
#define TWOSCALE_BOXFILTER_H

#include "headers.h"

class Filter
{
public:
    Filter() = default;
    Filter(int r, int wid, int hei);
    ~Filter();

    void boxfilterGlo(float *d_imgOut, float *d_imgIn, int wid, int hei, float *d_filter, int filterR);    // boxfilter based on global memory
    void boxfilterSha(float *d_imgOut, float *d_imgIn, int wid, int hei, float *d_filter, int filterR);    // boxfilter based on shared memory
    // the Texture memory based boxfitler is implemented on 2D Pitch memory,
    void boxfilterTex(float *d_imgOut, int wid, int hei, float *d_filter, int filterR);    // boxfilter based on texture memory
    void boxfilterAcc(float *d_imgOut, float *d_imgIn, int wid, int hei, float *d_filter, int filterR);    // boxfilter based on Separate Acccumulation
private:
    int rad_boxfilter_;
    float *h_img_filter_, *d_img_Pitch_filter_;
    float *d_temp_;
};

#endif //TWOSCALE_BOXFILTER_H
