#include <iostream>
#include "cassert"
#include "vector"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "cuda_runtime.h"
#include "cuda.h"

using namespace std;
using namespace cv;

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void
dataCopy(float *out, float *in, int row, int col)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx < col && idy < row)
    {
        int offset = idy * col + idx;
        out[offset] = in[offset];
    }
}

__global__ void
transposeNaive(float *out, float *in, int row, int col)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < col && idy < row)
    {
        int inIdx = idy * col + idx;
        int outIdx = idx * row + idy;
        out[outIdx] = in[inIdx];
    }
}

__global__ void
transposeShareMem(float *out, float *in, int row, int col)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx < col && idy < row)
    {
        int inIdx = idy * col + idx;
    }
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    Mat imgIn = imread("barbara.jpg", IMREAD_GRAYSCALE);
    assert(!imgIn.empty());
    imgIn.convertTo(imgIn, CV_32F, 1.0/255);

    Mat imgOut = Mat::zeros(imgIn.size(), CV_32F);

    const float *imgInP = (float *)imgIn.data;
    float *imgOutP = (float *)imgOut.data;
    const int row = imgIn.rows;
    const int col = imgIn.cols;

    cudaError_t cudaState = cudaSuccess;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime = 0.0;

    float *imgInD, *imgOutD;
    size_t pitchSrc, pitchDst;
    cudaState = cudaMallocPitch((void **)&imgInD, &pitchSrc, col * sizeof(float), row);   // the width is in bytes
    assert(cudaState == cudaSuccess);
    cudaState = cudaMemcpy2D(imgInD, pitchSrc, imgInP, sizeof(float) * col, sizeof(float) * col, row, cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMallocPitch((void **)&imgOutD, &pitchDst, col * sizeof(float), row);
    assert(cudaState == cudaSuccess);

    dim3 threadPerBlock(TILE_DIM, TILE_DIM);
    dim3 blockPerGrid;
    blockPerGrid.x = (col + TILE_DIM - 1) / TILE_DIM;
    blockPerGrid.y = (row + TILE_DIM - 1) / TILE_DIM;

    cudaEventRecord(start);
    //dataCopy<<<blockPerGrid, threadPerBlock>>>(imgOutD, imgInD, row, col);              // 0.063648ms
    //transposeNaive<<<blockPerGrid, threadPerBlock>>>(imgOutD, imgInD, row, col);          // 0.196096ms

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Used Time: " << elapsedTime << " ms." << endl;


    cudaState = cudaMemcpy2D(imgOutP, sizeof(float) * col, imgOutD, pitchDst, sizeof(float) * col, row, cudaMemcpyDeviceToHost);
    assert(cudaState == cudaSuccess);

    imshow("Output", imgOut);
    waitKey(0);

    return 0;
}
