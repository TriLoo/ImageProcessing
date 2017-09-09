//
// Created by smher on 17-9-1.
//

#ifndef GFFFUSION_TWOSCALE_H
#define GFFFUSION_TWOSCALE_H

#include "boxfilter.h"

// Set the BFilter to virtual base class
class TScale : public virtual BFilter
{
public:
    TScale(int r):rad(r)
    {
        cudaMalloc((void **)&d_filter_, (2 * r + 1) * ( 2 * r + 1) * sizeof(float));
        //cout << "d_filter = " << d_filter_ << endl;
        cout << "TScale Initialize Success" << endl;
    }
    ~TScale()
    {
        cudaFree(d_filter_);
    }

    void createMeanfilter();

    // the input image is stored on 2D pitch
    void twoscale(float *d_outH, float *d_outL, const float *d_in, size_t pitch, int wid, int hei, int filterW);
    // the input image is stored on the global memory
    void twoscale(float *d_outH, float *d_outL, const float * d_in, int wid, int hei);
private:
    int rad;
    float *d_filter_;   // filter on device global memory
};

#endif //GFFFUSION_TWOSCALE_H
