#include "calHist.h"

/* ***** use global memory ***** */
/*
__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo)

{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while(i < size)
	{
		atomicAdd(&(histo[buffer[i]]), 1);
		i += stride;
	}
}
*/
/* ***** use shared memory ***** */
__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo)

{
	__shared__ unsigned int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while(i < size)
	{
		atomicAdd(&(temp[buffer[i]]), 1);
		i += stride;
	}

	__syncthreads();

	atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

void calHist(unsigned char *data, const int size, unsigned int *hist)
{
	// GPU time
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	unsigned char *dev_buffer;
	unsigned int *dev_hist;

	cudaMalloc((void **)&dev_buffer, size * sizeof(unsigned char));
	cudaMemcpy(dev_buffer, data, size*sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&dev_hist, 256 * sizeof(unsigned int));
	cudaMemset(dev_hist, 0, 256 * sizeof(unsigned int));

	dim3 grid = GRID;
	dim3 block = BLOCK;

	histo_kernel<<<grid, block>>>(dev_buffer, size, dev_hist);

	// copy data from device back to host
	cudaMemcpy(hist, dev_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop,0);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout <<  "GPU time : " << elapsedTime << " ms" << endl;

	// free device memory
	cudaFree(dev_buffer);
	cudaFree(dev_hist);
}
