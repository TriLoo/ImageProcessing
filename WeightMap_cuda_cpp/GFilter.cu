#include "GFilter.h"

//#define INDEXOFFSET

GFilter::GFilter(int wid, int hei)
{
    cudaCheckErrors(cudaMalloc((void **)&d_tempA_, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempB_, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempC_, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempD_, sizeof(float) * wid * hei));
}

GFilter::~GFilter()
{
    if(d_tempA_)
        cudaFree(d_tempA_);
    if(d_tempB_)
        cudaFree(d_tempB_);
    if(d_tempC_)
        cudaFree(d_tempC_);
    if(d_tempD_)
        cudaFree(d_tempD_);
}

__global__ void elemwiseMultSame(float *out, float *in, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    float val = in[offset];
    out[offset] = __fmul_rd(val, val);
}

__global__ void elemwiseMult(float *out, float *in, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    out[offset] *= in[offset];
}

__global__ void elemwiseMult(float *out, float *inA, float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    out[offset] = inA[offset] * inB[offset];
}

__global__ void varianceKernel(float *out, float *in, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    float val = in[offset];
    out[offset] -= __fmul_rd(val, val);
}

__global__ void covarianceKernel(float *out, float *inA, float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    out[offset] -= __fmul_rd(inA[offset], inB[offset]);
}

__global__ void elemwiseDiv(float *out, float *in, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;o

    int offset = idx + idy * wid;

    //out[offset] /= in[offset];
    out[offset] = __fdiv_rd(out[offset], in[offset]);
}

__global__ void calculateA_kernel(float *out, float *in, int wid, int hei, double eps)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    float val = in[offset] + eps;
    out[offset] = __fdiv_rd(out[offset], val);
}

__global__ void calculateB_kernel(float *out, float *inA, float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    //out[offset] -= inA[offset] * inB[offset];
    out[offset] -= __fmul_rd(inA[offset], inB[offset]);
}

__global__ void calculateQ_kernel(float *out, float *inA, float *inB, float *inC, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idx + idy * wid;

    out[offset] = __fmul_rd(inA[offset], inB[offset]) + inC[offset];
}

void GFilter::guidedfilter(float *d_imgOut, float *d_imgInI, float *d_imgInP, int wid, int hei, int rad, double eps)
{
    // declare four streams to enable concurrence kernel launch.
    cudaStream_t stream[2];
    for(int i = 0; i < 2; ++i)
        cudaCheckErrors(cudaStreamCreate(&stream[i]));

    // prepare the needed configuration
    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + BLOCKSIZE - 1) / BLOCKSIZE;
    blockPerGrid.y = (hei + BLOCKSIZE - 1) / BLOCKSIZE;

    // define the Box Filter
    BFilter bf(wid, hei);

    // Step 1 : calculate the variance of image I
    // d_tempB_ : the result of fmean(imgI, r);
    bf.boxfilter(d_tempB_, d_imgInI, wid, hei, rad);
    elemwiseMultSame<<<blockPerGrid, threadPerBlock, 0, stream[0]>>>(d_tempA_, d_imgInI, wid, hei);
    bf.boxfilter(d_tempC_, d_tempA_, wid, hei, rad);
    varianceKernel<<<blockPerGrid, threadPerBlock, 0, stream[1]>>>(d_tempC_, d_tempB_, wid, hei);

    cout << "Step 1 : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    // Step 2 : calculate the covariance I-P
    elemwiseMult<<<blockPerGrid, threadPerBlock, 0, stream[0]>>>(d_tempA_, d_imgInI, d_imgInP, wid, hei);
    bf.boxfilter(d_tempD_, d_tempA_, wid, hei, rad);
    bf.boxfilter(d_tempA_, d_imgInP, wid, hei, rad);
    covarianceKernel<<<blockPerGrid, threadPerBlock, 0, stream[1]>>>(d_tempD_, d_tempB_, d_tempA_, wid, hei);

    cout << "Step 2 : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    // Step 3 : calculate the a  &   b
    calculateA_kernel<<<blockPerGrid, threadPerBlock, 0, stream[0]>>>(d_tempD_, d_tempC_, wid, hei, eps);
    calculateB_kernel<<<blockPerGrid, threadPerBlock, 0, stream[1]>>>(d_tempA_, d_tempD_, d_tempB_, wid, hei);

    cout << "Step 3 : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    // Step 4 : calculate the mean of a  &  b
    bf.boxfilter(d_tempB_, d_tempD_, wid, hei, rad);
    bf.boxfilter(d_tempD_, d_tempA_, wid, hei, rad);

    cout << "Step 4 : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    // Step 5 : Get the final result q
    calculateQ_kernel<<<blockPerGrid, threadPerBlock, 0, stream[0]>>>(d_imgOut, d_tempB_, d_imgInI, d_tempD_, wid, hei);

    cout << "Step 5 : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cudaDeviceSynchronize();

    for(int i = 0; i < 2; ++i)
        cudaStreamDestroy(stream[i]);
}

void GFilter::guidedfilterTest(float *imgOut, float *imgInI, float *imgInP, int wid, int hei, int rad, double eps)
{
    float *d_imgInI, *d_imgInP, *d_imgOut;
    cudaEvent_t start, stop;

    cudaCheckErrors(cudaMalloc((void **)&d_imgInI, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgInP, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgOut, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemset(d_imgOut, 0, sizeof(float) * wid * hei));

    cudaCheckErrors(cudaMemcpy(d_imgInI, imgInI, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_imgInP, imgInP, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    guidedfilter(d_imgOut, d_imgInI, d_imgInP, wid, hei, rad, eps);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Only Guided Filter on GPU: " << elapsedTime << " ms" << endl;

    cudaCheckErrors(cudaMemcpy(imgOut, d_imgOut, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
}

