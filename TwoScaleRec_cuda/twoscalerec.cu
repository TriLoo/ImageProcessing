#include "twoscalerec.h"

TwoScaleRec::TwoScaleRec(int wid, int hei)
{
    cudaError_t cudaState = cudaSuccess;
    cudaState = cudaMalloc((void **)&d_tempA_, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_tempB_, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_tempC_, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_tempD_, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_tempT_, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
}

TwoScaleRec::~TwoScaleRec()
{
    if(d_tempA_)
        cudaFree(d_tempA_);
    if(d_tempB_)
        cudaFree(d_tempB_);
    if(d_tempC_)
        cudaFree(d_tempC_);
    if(d_tempD_)
        cudaFree(d_tempD_);
    if(d_tempT_)
        cudaFree(d_tempT_);
}

__host__ __device__ float multadd_device(float inA, float inB, float inC, float inD)
{
    return inA * inB + inC * inD;
}

__global__ void multadd_kernel(float *out, float *inA, float *inB, float *inC, float *inD, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    out[offset] = multadd_device(inA[offset], inB[offset], inC[offset], inD[offset]);
}

__global__ void multadd_kernel(float *out, float *inA, float *inB, float *inC, float *inD, float *inT, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    out[offset] = multadd_device(inA[offset], inB[offset], inC[offset], inD[offset]) + inT[offset];
}


void TwoScaleRec::multadd(float *d_Out, float *inA, float *inB, float *inC, float *inD, int wid, int hei)
{
    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    multadd_kernel<<<blockPerGrid, threadPerBlock>>>(d_Out, inA, inB, inC, inD, wid, hei);

    cout << "Multiply and Addition : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;
}

void TwoScaleRec::twoscalerec(float *out, float *d_imgInA, float *d_imgInB, int wid, int hei)
{

}

void TwoScaleRec::twoscalerecTest(float *out, vector<float *> &imgInBase, vector<float *> &imgInDetail, int wid,
                                  int hei)
{
}
