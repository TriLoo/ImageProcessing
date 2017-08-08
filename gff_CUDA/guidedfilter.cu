#include "guidedfilter.h"

__global__ void cumSumCol(float *imgIn, float *imgOut, int row, int col)
{
    //__shared__ float *cacheIn[blockDim.x * blockDim.y];
    //__shared__ float *cacheOut[blockDim.x * blockDim.y];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int index = idy * row + idx;
    //int cacheIdx = threadIdx.y * blockDim.x + blockIdx.x;

    /*
    cacheIn[cacheIdx] = imgIn[index];
    cacheOut[cacheIdx] = 0;
    __syncthreads();
    */

    while()
    {
        imgOut[]
    }



}

__global__ void cumSumRow(float *imgIn, float *imgOut, int row, int col)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

}

void GFilter::boxfilter(cv::Mat &imgIn)
{

}

void GFilter::guidedfilter(cv::Mat &imgI, cv::Mat &imgP)
{

}