#include "twoscale.h"

// function : d_imgOut = d_imgIn - d_imgOut
// the input B and output is the same : d_imgOut
__global__ void elemwiseSub_kernel(float *d_imgOut, float *d_imgIn, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    d_imgOut[INDX(idy, idx, wid)] -= d_imgIn[INDX(idy, idx, wid)];
}

void TwoScale::twoscale(float *d_imgOut, float *d_imgIn, int wid, int hei, int rad)
{
    boxfilterAcc(d_imgOut, d_imgIn, wid, hei, nullptr, rad);
}