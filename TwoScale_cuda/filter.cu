#include "filter.h"

// do boxfilter on separable two dimension accumulation
// process row
__device__ void d_boxfilter_x(float *d_imgOut, float *d_imgIn, int wid, int hei, int filterR)
{
    float scale = 1.0f / ((filterR << 1) + 1);         // the width of filter

    float t = 0.f;

    // do the left edge
    t = d_imgIn[0] * filterR;
    for(int x = 0; x < (filterR + 1); x++)
        t += d_imgIn[x];

    d_imgOut[0] = __fmul_rd(t, scale);

    for(int x = 1; x < (filterR + 1); x++)
    {
        t += d_imgIn[x + filterR];
        t -= d_imgIn[0];
        d_imgOut[x] = __fmul_rd(t, scale);
    }

    // main loop
    for(int x = (filterR + 1); x < (wid - filterR); x++)
    {
        t += d_imgIn[x + filterR];
        t -= d_imgIn[x - 1 - filterR];
        d_imgOut[x] = __fmul_rd(t, scale);
    }

    // do right edge
    for(int x = wid - filterR; x < wid; x++)
    {
        t += d_imgIn[wid - 1];
        t -= d_imgIn[wid - filterR - 1];

        d_imgOut[x] = __fmul_rd(t, scale);
    }
}
// process col
__device__ void d_boxfilter_y(float *d_imgOut, float *d_imgIn, int wid, int hei, int filterR)
{
    float scale = 1.0f / ((filterR << 1) + 1);
    float t = 0.f;

    // do upper edge
    t = d_imgIn[0] * filterR;

    for(int y = 0; y < (filterR + 1); y++)
        t += d_imgIn[y * wid];

    d_imgOut[0] = __fmul_rd(t, scale);

    for(int y = 1; y < (filterR + 1); y++)
    {
        t += d_imgIn[(y + filterR) * wid];
        t -= d_imgIn[0];
        d_imgOut[y * wid] = __fmul_rd(t, scale);
    }

    // main loop
    for(int y = (filterR + 1); y < (hei - filterR); y++)
    {
        t += d_imgIn[(y + filterR) * wid];
        t -= d_imgIn[(y - filterR - 1) * wid];
        d_imgOut[y] = __fmul_rd(t, scale);
    }

    // do the bottom edge
    for(int y = (hei - filterR); y < hei; y++)
    {
        t += d_imgIn[(hei - 1) * wid];
        t -= d_imgIn[(y - filterR - 1) * wid];
        d_imgOut[y * wid] = __fmul_rd(t, scale);
    }
}

// global function
// row part
__global__ void d_boxfilter_x_global(float *d_imgOut, float *d_imgIn, int wid, int hei, int filterR)
{
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    d_boxfilter_x(&d_imgOut[y * wid], &d_imgIn[y * wid], wid, hei, filterR);
}

// global function
// col part
__global__ void d_boxfilter_y_global(float *d_imgOut, float *d_imgIn, int wid, int hei, int filterR)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    d_boxfilter_y(&d_imgOut[x], &d_imgIn[x], wid, hei, filterR);
}

void Filter::boxfilterAcc(float *d_imgOut, float *d_imgIn, int wid, int hei, float *d_filter, int filterR)
{
    int nthreads = 512;

    dim3 threadPerBlock;
    threadPerBlock.x = 512;
    threadPerBlock.y = 1;
    dim3 blockPerGrid;
    blockPerGrid.x = (hei + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = 1;

    d_boxfilter_x_global<<<blockPerGrid, threadPerBlock>>>(d_temp_, d_imgIn, wid, hei, filterR);
    cout << "Boxfilter Row Process : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = 1;
    d_boxfilter_y_global<<<blockPerGrid, threadPerBlock>>>(d_imgOut, d_temp_, wid, hei, filterR);
    cout << "Boxfilter Col Process : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cudaCheckError(cudaDeviceSynchronize());
}

Filter::Filter(int r, int wid, int hei)
{
    rad_boxfilter_ = r;
    cudaCheckError(cudaMalloc((void **)&d_temp_, sizeof(float) * wid * hei));
    cudaCheckError(cudaMemset(d_temp_, 0, sizeof(float) * wid * hei));
}

Filter::~Filter()
{
    if(d_temp_)
        cudaFree(d_temp_);
}

