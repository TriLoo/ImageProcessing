#include "twoscale.h"

#define INDX(r, c, w) ((r) * (w) + (c))

// the input image in stored on global memory
// A : input image, B : the low-pass coefficients
__global__ void tscaleGlo(float *d_out, const float *d_inA, float *d_inB, int wid, int hei)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if(idx > wid || idy > hei)
        return;

    int offset = INDX(idy, idx, wid);

    d_out[offset] = d_inA[offset] - d_inB[offset];
}

// the input image in stored on 2D Pitch
// A : input iamge, B : the low-pass coefficients
__global__ void tscaleTex(float *d_out, const float *d_inA, float *d_inB, size_t pitch, int wid, int hei)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if(idx > wid || idy > hei)
        return;

    int offset = INDX(idy, idx, wid);
    int offsetA = INDX(idy, idx, pitch / sizeof(float));   // the width of image is pitch / sizeof(float)

    d_out[offset] = d_inA[offsetA] - d_inB[offset];
}

// generate mean filter used in boxfilter
void TScale::createMeanfilter()
{
    cudaError_t cudaState = cudaSuccess;

    int filterWidth = 2 * rad + 1;
    const int filterSize = filterWidth * filterWidth;
    float *filter = new float [filterSize];

    for(int i = 0; i < filterSize; ++i)
        filter[i] = 1 / filterSize;

    // copy data from host to device
    cudaState = cudaMemcpy((void **)&d_filter_, filter, sizeof(float) * filterSize, cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);
}

// launch two-scale decomposing of input image
// the input image is stored in global memory
// d_outH : high-pass coefficients, d_outL : low-pass coefficients
// d_in   : the input image
void TScale::twoscale(float *d_outH, float *d_outL, const float *d_in, int wid, int hei)
{
    int filterW = 2 * rad + 1;
    cudaError_t cudaState = cudaSuccess;
    boxfilter(d_outL, d_in, wid, hei,  d_filter_, filterW);

    // launch kernel to obtain the finnal results
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    tscaleGlo<<<blockPerGrid, threadPerBlock>>>(d_outH, d_in, d_outL, wid, hei);

    cudaState = cudaGetLastError();

    if(cudaState != cudaSuccess)
    {
        cout << "Function twoscale decomposing failed due to : " ;
        cout << cudaGetErrorString(cudaState) << endl;
    }

}

/*
// the input image is stored in 2D pitch
void TScale::twoscale(float *d_outH, float *d_outL, const float *d_in, size_t pitch, int wid, int hei, int filterW)
{
    cudaError_t cudaState = cudaSuccess;

     boxfilter(d_outL, d_in, pitch, wid, hei, d_filter_,filterW);

    // launch kernel to obtain the finnal results
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    //tscaleTex<<<blockPerGrid, threadPerBlock>>>(d_outH, d_in, d_outL, wid, hei);
    tscaleTex<<<blockPerGrid, threadPerBlock>>>(d_outH, d_in, d_outL, pitch, wid, hei);

    cudaState = cudaGetLastError();

    if(cudaState != cudaSuccess)
    {
        cout << "Function twoscale decomposing failed due to : " ;
        cout << cudaGetErrorString(cudaState) << endl;
    }
}
*/
