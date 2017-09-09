#include "tworec.h"

// matrix-matrix multiple based on global memory
__global__ void elemmult(float *out, float *inA, float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;    // col number
    int idy = threadIdx.y + blockDim.y * blockIdx.y;    // row number

    if(idx > wid || idy > hei)
        return ;

    int offset = idy * wid + idx;

    // use "+=" because of A .* B + C .* D
    // so the result of second calling is just the final needed result
    out[offset] += inA[offset] * inB[offset];
}

__global__ void elem_add(float *d_out, float *inA, float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx > wid || idy > hei)
        return ;

    int offset = idy * wid + idx;
    d_out[offset] = inA[offset] + inB[offset];
}

// get the result coefficients based on element-wise multiple, don't need any cublas APIs
void TRec::getcoeff(float *d_out, float *d_inA, float *d_inB, float *d_inA_W, float *d_inB_W, int wid, int hei)
{
    cudaError_t cudaState = cudaSuccess;

    // memset d_out to 0
    cudaState = cudaMemset(d_out, 0, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    /*
    // create cublas handle
    cublasStatus_t cudaBlasState = CUBLAS_STATUS_SUCCESS;
    cublasHandle_t handle;
    cudaBlasState = cublasCreate(&handle);
    assert(cudaBlasState == CUBLAS_STATUS_SUCCESS);


    // TODO : Caution the transpose when using cublas
    // use cublas level - 3 : matrix - matrix operations
    // calculate the d_out = d_inA .* d_inA_W;
    // CUBLAS_OP_N : Don'd perform transform on input matrix
    // leading dimesion : the rows(for col-major storage) or cols(for row-major storage) for input matrix
    // no matter the input matrix is transposed or not ! ! !
    //cudaBlasState = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hei, wid, wid, 1, d_inA, hei, d_inB, hei, hei, 0, d_out, hei);
    */

    // launch the kernels
    // calculate the d_out = d_inA .* d_inA_W;
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;
    // calculate the d_out = d_inA .* d_inA_W;
    elemmult<<<blockPerGrid, threadPerBlock>>>(d_out, d_inA, d_inA_W, wid, hei);

    // calculate the d_out += d_inB .* d_inB_W;
    elemmult<<<blockPerGrid, threadPerBlock>>>(d_out, d_inB, d_inB_W, wid, hei);

    // assure the kernel run successfully
    cudaState = cudaGetLastError();
    if(cudaState != cudaSuccess)
        cout << "Error in function get the coefficients : " << cudaGetErrorString(cudaState) << endl;

    // destroy cublas by calling cublasDestroy
    //cudaBlasState = cublasDestroy(handle);
}

// get the final fusion result by add two layers
void TRec::twoscalerec(float *d_out, float *d_inA, float *d_inB, int wid, int hei)
{
    cudaError_t cudaState = cudaSuccess;

    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    elem_add<<<blockPerGrid, threadPerBlock>>>(d_out, d_inA, d_inB, wid, hei);

    cudaState = cudaGetLastError();

    if(cudaState != cudaSuccess)
        cout << "In function two scale restore : " << cudaGetErrorString(cudaState) << endl;
}


