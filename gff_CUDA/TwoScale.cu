#include "TwoScale.h"

TwoScale::~TwoScale()
{
}

__global__ void elemwiseSub_kernel(float *out, float *inA, float *inB, int wid, int hei)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	if(idx >= wid || idy >= hei)
		return ;

	int offset = idy * wid + idx;
	out[offset] = inA[offset] - inB[offset];
}


void TwoScale::twoscale(float *d_imgOutA, float *d_imgOutB, float *d_imgIn, const int wid, const int hei,
                        const int filterR)
{
    // * d_imgOutB is the low-pass part
    boxfilter(d_imgOutB, d_imgIn, wid, hei, filterR);
    // get the high pass coefficients
	dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 blockPerGrid;
	blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

	elemwiseSub_kernel<<<blockPerGrid, threadPerBlock>>>(d_imgOutA, d_imgIn, d_imgOutB, wid, hei);
	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;
}

void TwoScale::twoscaleTest(float *imgOutA, float *imgOutB, float *imgIn, const int wid, const int hei,
                            const int filterR)
{
    float *d_imgIn, *d_imgOutA, *d_imgOutB;
	cudaCheckErrors(cudaMalloc((void **)&d_imgIn, sizeof(float) * wid * hei));
	cudaCheckErrors(cudaMemcpy(d_imgIn, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));
	cudaCheckErrors(cudaMalloc((void **)&d_imgOutA, sizeof(float) * wid * hei));
	cudaCheckErrors(cudaMalloc((void **)&d_imgOutB, sizeof(float) * wid * hei));

    twoscale(d_imgOutA, d_imgOutB, d_imgIn, wid, hei, filterR);

    cudaCheckErrors(cudaMemcpy(imgOutB, d_imgOutB, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
	cudaCheckErrors(cudaMemcpy(imgOutA, d_imgOutA, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
}
