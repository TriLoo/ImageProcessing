#include "guidedfilter.h"

// two same matrix multiple based on global memory
__global__ void elemmultSame(float *out, float *in, int wid, int hei)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;    // col number
    int idy = threadIdx.y + blockDim.y * blockIdx.y;    // row number

    if(idx > wid || idy > hei)
        return ;

    int offset = idy * wid + idx;

    // use "+=" because of A .* B + C .* D
    // so the result of second calling is just the final needed result
    float val = in[offset];
    out[offset] = val * val;
}

// matrix-matrix multiple based on global memory
__global__ void elem_mult(float *out, float *inA, float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;    // col number
    int idy = threadIdx.y + blockDim.y * blockIdx.y;    // row number

    if(idx > wid || idy > hei)
        return ;

    int offset = idy * wid + idx;

    out[offset] = inA[offset] * inB[offset];
}

// matrix-matrix element-wise add based on global memory
__global__ void elemadd(float *d_out, float *inA, float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx > wid || idy > hei)
        return ;

    int offset = idy * wid + idx;
    d_out[offset] = inA[offset] + inB[offset];
}

// variance calculation for : varI = corrI - meanI .* meanI
// d_inA : the corrI stored in d_tempC; d_inB : the meanI stored in d_tempA
// here, d_inA and d_out is the same matrix
//__global__ void elemvar(float *d_out, float *d_inA, float *d_inB, int wid, int hei)
__global__ void elemvar(float *d_out, float *d_inB, int wid, int hei)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if(idx > wid || idy > hei)
        return ;

    int offset = idy * wid + idx;
    float val = 0.f;
    val = d_inB[offset];
    val *= val;   // val = meanI .* meanI

    d_out[offset] -= val;   // d_out = d_inA are both the corrI to use less device memory
}

// calculation convariance based on : covIp = corrIp - meanI .* meanP
// d_inA : the corrIp stored in tempD, d_inB : meanI stored in tempA, d_inC : meanP stored in tempB
// d_out is same with d_inA i.e. tempD, so d_inA isn't displayed here
//__global__ void elemcov(float *d_out, float *d_inA, float *d_inB, float *d_inC, int wid, int hei)
__global__ void elemcov(float *d_out, float *d_inB, float *d_inC, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx > wid || idy > hei)
        return ;

    int offset = idy * wid + idx;
    float val = 0.f;

    val = d_inB[offset] * d_inC[offset];
    d_out[offset] -= val;   // same as : covIp = corrIp - meanI .* meanP
}

// calculate 'a' based on a = covIp ./(varI + eps)
// varI is stored in  tempC for now, covIp is stored in tempD for now
// the result is stored back to tempD
__global__ void elemdiv(float *d_out, float *d_inA, float e, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx > wid || idy > hei)
        return ;

    int offset = idy * wid + idx;
    float val = 0.f;

    val = d_inA[offset] + e;  // e is 'eps'
    d_out[offset] /= val;
}

// calculate the final result based on q = meanA .* I + meanB
// d_inA : meanA
// d_inB : 'I'
// d_inC : meanB
__global__ void elemres(float *d_out, float *d_inA, float *d_inB, float *d_inC, int wid, int hei)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if(idx > wid || idy > hei)
        return ;

    int offset = idy * wid + idx;

    d_out[offset] = d_inA[offset] * d_inB[offset] + d_inC[offset];
}

void GFilter::createGufilter()
{
    cudaError_t cudaState = cudaSuccess;
    int filterWidth = 2 * rad_ + 1;
    int filterSize = filterWidth * filterWidth;
    float *filter = new float [filterSize];

    // mean filter
    for(int i = 0; i < filterSize; ++i)
        filter[i] = 1 / filterSize;

    // copy filter from host to device
    cudaState = cudaMalloc((void **)&d_guifilter_, sizeof(float) * filterSize);
    assert(cudaState == cudaSuccess);

    cudaState = cudaMemcpy(d_guifilter_, filter, sizeof(float) * filterSize, cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);

    cudaDeviceSynchronize();

    cout << "Guided filter Generated" << endl;

    delete [] filter;
}

// for more details : see < Fast Guided Filter>

void GFilter::guidedfilter(float *d_out, float *d_inI, float *d_inP, int wid, int hei)
{
    cudaError_t cudaState = cudaSuccess;

    // prepare four memories
    float *d_tempA, *d_tempB, *d_tempC, *d_tempD;
    cudaState = cudaMalloc((void **)&d_tempA, sizeof(float) * wid * hei);
    cudaState = cudaMalloc((void **)&d_tempB, sizeof(float) * wid * hei);
    cudaState = cudaMalloc((void **)&d_tempC, sizeof(float) * wid * hei);
    cudaState = cudaMalloc((void **)&d_tempD, sizeof(float) * wid * hei);

    int filterWidth = 2 * rad_ + 1;

    createGufilter();

    // d_tempA = meanI : the result of fmean(I, r);
    boxfilter(d_tempA, d_inI, wid, hei, d_guifilter_, filterWidth);
    // d_tempB = meanP : the result of fmean(p, r);
    boxfilter(d_tempB, d_inP, wid, hei, d_guifilter_, filterWidth);

    // launch kernel function to calculate the element-wise multiple
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;
    // tempC = I .* I
    elemmultSame<<<blockPerGrid, threadPerBlock>>>(d_tempC, d_inI, wid, hei);
    // tempC = fmean(I .* I, r) = corrI
    boxfilter(d_tempC, d_tempC, wid, hei, d_guifilter_, filterWidth);

    // tempD = I .* p
    elem_mult<<<blockPerGrid, threadPerBlock>>>(d_tempD, d_inI, d_inP, wid, hei);
    // tempD = fmean( I .* p, r) = corrIp
    boxfilter(d_tempD, d_tempD, wid, hei, d_guifilter_, filterWidth);


    // calculate the variance by : varI = corrI - meanI ./* meanI
    // based on elemvar kernel function
    // result is stored in tempC again
    elemvar<<<blockPerGrid, threadPerBlock>>>(d_tempC, d_tempA, wid, hei);

    // calculate the covariance of Ip based on : covIp = corrIp - meanI .* meanP, meanI : d_tempA, meanP : d_tempB
    // based on elemcov kernel function
    // result is stored in tempD again
    elemcov<<<blockPerGrid, threadPerBlock>>>(d_tempD, d_tempA, d_tempB, wid, hei);

    // calculate a & b
    // a = covIp ./(varI + eps);   // result is stored in covIp i.e. temp_D
    // b= meanP - a .* meanI using elemcov kernel function and result is stored back to d_tempB
    // for now : tempA - meanI; tempB - b; tempC - varI, tempD - a
    // calculate 'a' based on elemdiv kernel function
    elemdiv<<<blockPerGrid, threadPerBlock>>>(d_tempD, d_tempC, eps_, wid, hei);
    // calculate 'b' based on elemcov kernel function
    elemcov<<<blockPerGrid, threadPerBlock>>>(d_tempB, d_tempD, d_tempA, wid, hei);

    // calculate meanA, meanB based on boxfilter
    boxfilter(d_tempA, d_tempD, wid, hei, d_guifilter_, filterWidth);  // d_tempA is storing meanA, d_tempD is 'a'
    boxfilter(d_tempB, d_tempB, wid, hei, d_guifilter_, filterWidth);  // d_tempB is storing meanB, d_tempB is 'b'

    // get the guided filter result based on : q = meanA .* I + meanB
    // based on elemres(result) kernel function
    // result is stored in d_out
    elemres<<<blockPerGrid, threadPerBlock>>>(d_out, d_tempA, d_inI, d_tempB, wid, hei);

    cudaState = cudaGetLastError();

    if(cudaState != cudaSuccess)
        cout << "In function guided filter : " << cudaGetErrorString(cudaState) << endl;

    // free temp memories on device
    cudaFree(d_tempA);
    cudaFree(d_tempB);
    cudaFree(d_tempC);
    cudaFree(d_tempD);
}
