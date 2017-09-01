#include "weightmap.h"

/*
#define LapFilterWid 3
#define LapFilterSize (LapFilterWid * LapFilterWid)
#define GauFilterRad 5
#define GauFilterWid (2 * GauFilterRad + 1)  // 11 = 2 * 5 + 1
#define GauFilterSize (GauFilterWid * GauFilterWid)
#define SIGMA 0.9
#define PI 3.14159
*/

// r : row, c : col, w : width
#define INDX(r, c, w) ((r * w) + c)

__device__ int clamp(int a, int b, int c)
{
    int temp = a > b ? a : b;
    temp = temp < c ? temp : c;

    return temp;
}

// calculate the corresponding filter, the calculation is based on shared memory
__global__ void filter(float *d_imgOut, const float *d_imgIn, int hei, int wid, const float * __restrict__ d_filter, const int filterWidth)
{
    const unsigned int x0 = threadIdx.x;
    const unsigned int y0 = threadIdx.y;

    // determine the thread index
    int tx = blockDim.x * blockIdx.x + x0;
    int ty = blockDim.y * blockIdx.y + y0;

    if(tx >= hei || ty >= wid)
        return ;

    const int filterRadius = filterWidth / 2;
    float val = 0.f;
    const int extWid = blockDim.x + filterWidth - 1;

    // copy data from global memory to shared memory which is dynamic allocated
    extern __shared__ float shareImg[];

    int fx, fy;   // filter index

    // case 1: upper left corner
    fx = tx - filterRadius;            // the width
    fy = ty - filterRadius;            // the height
    if(fx < 0 || fy < 0)
        shareImg[y0 * extWid + x0] = 0;
    else
        shareImg[y0 * extWid + x0] = d_imgIn[fy * wid + fx];

    // case 2: upper right
    fx = tx + filterRadius;
    fy = ty - filterRadius;
    if(fx > wid - 1|| fy < 0)
        shareImg[y0 * extWid + x0 + filterWidth - 1] = 0;
    else
        shareImg[y0 * extWid + x0 + filterWidth - 1] = d_imgIn[fy * wid + fx];

    // case 3: lower left
    fx = tx - filterRadius;
    fy = ty + filterRadius;
    if(fx < 0 || fy > hei - 1)
        shareImg[(y0 + filterWidth - 1) * extWid + x0] = 0;
    else
        shareImg[(y0 + filterWidth - 1) * extWid + x0] = d_imgIn[fy * wid + fx];

    // case 4 : lower right
    fx = tx + filterWidth;
    fy = ty + filterWidth;
    if(fx > wid - 1 || fy > hei - 1)
        shareImg[(y0+filterWidth - 1) * extWid + x0 + filterWidth - 1]  = 0;
    else
        shareImg[(y0+filterWidth - 1) * extWid + x0 + filterWidth - 1]  = d_imgIn[fy * wid + fx];

    // synthreads for share memory
    __syncthreads();

    for(int fr = 0; fr < filterWidth; ++fr)
        for(int fc = 0; fc <filterWidth; ++fc)
            val += shareImg[(y0 + fr) * extWid + fc + x0] * d_filter[INDX(fr, fc, filterWidth)];

    d_imgOut[INDX(ty, tx, wid)] = val >= 0 ? val : -val;
}


//void WeightMap::laplafilter(float *d_imgOut, float *d_imgIn, int height, int width, const float * __restrict__ d_filter, const int rad)
void WeightMap::laplafilter(float *d_imgOut, const float *d_imgIn, const int height, const int width,
                            const float *d_filter, const int rad)
{
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (width + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (height + threadPerBlock.y - 1) / threadPerBlock.y;
    const int filterWidth = 2 * rad + 1;
    size_t shareSize = (threadPerBlock.x + 2 * rad) * (threadPerBlock.y + 2 * rad);
    filter<<<blockPerGrid, threadPerBlock, shareSize>>>(d_imgOut, d_imgIn, height, width, d_filter, filterWidth);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        cout << cudaGetErrorString(err) << endl;
}

//void WeightMap::gaussfilter(float *d_imgOut, float *d_imgIn, int height, int width, const float * __restrict__ d_filter, const int rad)
void WeightMap::gaussfilter(float *d_imgOut, const float *d_imgIn, const int height, const int width,
                            const float *d_filter, const int rad)
{
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (width + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (height + threadPerBlock.y - 1) / threadPerBlock.y;
    const int filterWidth = 2 * rad + 1;
    int shareSize = (threadPerBlock.x + 2 * rad) * (threadPerBlock.y + 2 * rad);
    filter<<<blockPerGrid, threadPerBlock, shareSize>>>(d_imgOut, d_imgIn, height, width, d_filter, filterWidth);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        cout << cudaGetErrorString(err) << endl;
}

