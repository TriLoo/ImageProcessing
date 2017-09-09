//
// Created by smher on 17-9-1.
//

#ifndef GFFFUSION_TWOREC_H
#define GFFFUSION_TWOREC_H

#include "twoscale.h"
#include "weightmap.h"

// include cublas for matrix multiple
#include <cublas_v2.h>

class TRec
{
public:
    TRec() = default;
    ~TRec()  = default;

    // get the final low-pass  &  high-pass coefficients
    // d_inA : low-pass coefficients of image A get from two scale decomposing
    // d_inB : low-pass coefficients of image B get from two scale decomposing
    // d_inA_W : the weight map of image A
    // d_inB_W : the weight map of image B
    // coefficients = d_inA .* d_inA_W + d_inB .* d_inB_W;   // based on cublas for matrix multiple
    void getcoeff(float *d_out, float *d_inA, float *d_inB, float *d_inA_W, float *d_inB_W, int wid, int hei);

    // get the high-pass coefficients
    // see the low-pass for detail information
    //void highpass(flaot *d_out, float *d_inA, float *d_inB, float *d_inA_W, float *d_inB_W, int wid, int hei);

    // combine the low-pass & high-pass coefficients to obtain the final fusion result
    void twoscalerec(float *d_out, float *d_inA, float *d_inB, int wid, int hei);
private:
};

#endif //GFFFUSION_TWOREC_H
