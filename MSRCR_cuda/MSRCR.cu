#include "MSRCR.h"

#define BLOCKSIZE 16

MSRCR::~MSRCR()
{

}

__global__  void HF_Enhancer_kernel(float *d_out, float *d_in, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    //if(idx >= (wid >> 2) || idy >= hei)
    if(idx >= wid || idy >= hei)
        return ;

    // do four calculations per thread
#pragma unloop
    for(int i = 0; i < 4; ++i)   // column direction
    {
        if(idx + i >= wid)
            return;
#pragma unloop
        for(int j = 0; j < 4; ++j)   // row direction
        {
            if(idy + j >= hei)
                return;
            int offset = (j + idy) * wid + (idx + i);

            d_out[offset] = expf(logf(d_out[offset]) - logf(d_in[offset]));
        }
    }

    // do four calculations per thread, but reduce the memory coalescing
    /*
    for(int i = blockDim.x * gridDim.x; i > 0; i = i >> 2)   // divide 4 every time
    {
    }
    */
}

void MSRCR::High_Frequency_Enhancer(float *d_out, float *d_in, int wid, int hei)
{
    //dim3 threadPerBlock(((wid >> 2) + BLOCKSIZE - 1)/ BLOCKSIZE, ((hei >> 2) + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = ((wid >> 2) + BLOCKSIZE - 1) / BLOCKSIZE;
    blockPerGrid.y = ((hei >> 2) + BLOCKSIZE - 1) / BLOCKSIZE;

    HF_Enhancer_kernel<<<blockPerGrid, threadPerBlock>>>(d_out, d_in, wid, hei);
}

void MSRCR::MSR(float *d_out, float *d_in, int wid, int hei, double sigma)
{

}

void MSRCR::histEqu(float *d_out, float *d_in, int wid, int hei)
{

}




