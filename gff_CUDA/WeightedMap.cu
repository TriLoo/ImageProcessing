#include "WeightedMap.h"
#define GaussW 11

WMap::~WMap()
{

}

// do absolute laplacian filter based on shared memory
__global__ void laplacianAbs_kernel(float *out, float *in, int wid, int hei, const float * __restrict__ filter, int lr)
{
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    extern __shared__ float shareMem[];

    //int TILEW = blockDim.x + 2 * lr;
    int TILEW = BLOCKSIZE + 2 * lr;

    int x, y;
    // copy data from global memory to shared memory, zero extends the border
    // case 1 : upper left
    x = idx - lr;
    y = idy - lr;
    if(x < 0 || y < 0)
        shareMem[INDX(y0, x0, TILEW)] = 0;
    else
        shareMem[INDX(y0, x0, TILEW)] = in[INDX(y, x, wid)];

    // case 2 : upper right
    x = idx + lr;
    y = idy - lr;
    if(x >= wid || y < 0)
        shareMem[INDX(y0, x0 + 2 * lr, TILEW)] = 0;
    else
        shareMem[INDX(y0, x0 + 2 * lr, TILEW)] = in[INDX(y, x, wid)];

    // case 3 : lower left
    x = idx - lr;
    y = idy + lr;
    if(x < 0 || y >= hei)
        shareMem[INDX(y0 + 2 * lr, x0, TILEW)] = 0;
    else
        shareMem[INDX(y0 + 2 * lr, x0, TILEW)] = in[INDX(y, x, wid)];

    // case 4 : lower right
    x = idx + lr;
    y = idy + lr;
    if(x >= wid || y >= hei)
        shareMem[INDX(y0 + 2 * lr, x0 + 2 * lr, TILEW)] = 0;
    else
        shareMem[INDX(y0 + 2 * lr, x0 + 2 * lr, TILEW)] = in[INDX(y, x, wid)];

    __syncthreads();

    int lw = (lr << 1) + 1;
    float val = 0.f;
    for(int i = 0; i < lw; ++i)             // row
        for(int j = 0; j < lw; ++j)         // col
            val += shareMem[INDX(y0 + i, x0 + j, TILEW)] * filter[INDX(i, j, lw)];

    // obtain the absolute value
    out[INDX(idy, idx, wid)] = val >= 0 ? val : -val;
}

// do separable gaussian filter based on CUDA
__global__ void gaussfilterRow_kernel(float *out, float *in, int wid, int hei, const float * __restrict__ filter, int gr)
{
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;

    int idx = blockDim.x * blockIdx.x + x0;
    int idy = blockDim.y * blockIdx.y + y0;

    if(idx >= wid || idy >= hei)
        return ;

    extern __shared__ float shareMem[];

    int x, y;
    int TILEW = BLOCKSIZE + 2 * gr;
    // case 1 : left apron
    x = idx - gr;
    y = idy;
    if(x < 0)
        shareMem[INDX(y0, x0, TILEW)] = 0;
    else
        shareMem[INDX(y0, x0, TILEW)] = in[INDX(y, x, wid)];

    // case 2 : right apron
    x = idx + gr;
    y = idy;
    if(x >= wid)
        shareMem[INDX(y0, x0 + 2 * gr, TILEW)] = 0;
    else
        shareMem[INDX(y0, x0 + 2 * gr, TILEW)] = in[INDX(y, x, wid)];

    __syncthreads();

    float val = 0.f;
#pragma unrool
    for(int i = 0; i < GaussW; i++)
        val += __fmul_rd(shareMem[INDX(y0, x0 + i, TILEW)], filter[i]);

    out[INDX(idy, idx, wid)] = val;
}

__global__ void gaussfilterCol_kernel(float *out, float *in, int wid, int hei, float const * __restrict__ filter, int filterR)
{
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;

    int idx = blockDim.x * blockIdx.x + x0;
    int idy = blockDim.y * blockIdx.y + y0;

    if (idx >= wid || idy >= hei)
        return;

    //__shared__ float shareMem[ * BLOCKSIZE];
    extern __shared__ float shareMem[];

    int x, y;
    // case 1 : top apron
    y = idy - filterR;
    x = idx;
    if(y < 0)
        shareMem[INDX(y0, x0, BLOCKSIZE)] = 0;
    else
        shareMem[INDX(y0, x0, BLOCKSIZE)] = in[INDX(y, x, wid)];

    // case 2 : bottom apron
    y = idy + filterR;
    x = idx;
    if(y >= hei)
        shareMem[INDX(y0 + 2 * filterR, x0, BLOCKSIZE)] = 0;
    else
        shareMem[INDX(y0 + 2 * filterR, x0, BLOCKSIZE)] = in[INDX(y, x, wid)];

    __syncthreads();

    float val = 0.f;
#pragma unroll
    for(int i = 0; i < GaussW; ++i)
        //val += shareMem[INDX(y0 + i, x0, BLOCKSIZE)] * filter[i];
        val += __fmul_rd(shareMem[INDX(y0+i, x0, BLOCKSIZE)], filter[i]);

    out[INDX(idy, idx, wid)] = val;
}

// comparasion kernel
__global__ void comparison_kernel(float *outA, float *outB, float *inA, float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = idx + idy * wid;

    if(idx >= wid || idy >= hei)
        return ;

    int val = (inA[offset] >= inB[offset]) ? 1 : 0;
    outA[offset] = val;
    outB[offset] = 1 - val;
}
