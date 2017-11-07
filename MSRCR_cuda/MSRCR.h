//
// Created by smher on 17-11-7.
//

#ifndef MSRCR_MXRCR_H
#define MSRCR_MXRCR_H

#include "headers.h"

class MSRCR
{
public:
    MSRCR() = default;
    ~MSRCR();

    void High_Frequency_Enhancer(float *d_out, float *d_in, int wid, int hei);
    void MSR(float *d_out, float *d_in, int wid, int hei, double sigma);
    void histEqu(float *d_out, float *d_in, int wid, int hei);

private:
    void GetMinMax(float *d_in, int wid, int hei);
};

#endif //MSRCR_MXRCR_H
