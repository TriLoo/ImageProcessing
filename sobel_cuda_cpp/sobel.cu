#include "sobel.h"

__global__ void SobelKernel(int row, int col, float *imgIn, float *imgOut)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockIdx.x * gridDim.x;

	int lu, lm, lb, mu, mm, mb, ru, rm, rb;

	if(x < row && y < col)
	{
		int luIdx = offset - row - 1;
		int lmIdx = offset - 1;
		int lbIdx = offset + row - 1;

		int muIdx = offset - row;
		int mmIdx = offset;
		int mbIdx = offset + row;

		int ruIdx = offset - row + 1;
		int rmIdx = offset + 1;
		int rbIdx = offset + row + 1;


		if((x != 0) && (x != (row - 1)) && (y != 0) && (y != (col - 1)))
		{
			lu = imgIn[luIdx];
			mu = imgIn[muIdx];
			ru = imgIn[ruIdx];
			lm = imgIn[lmIdx];
			lb = imgIn[lbIdx];
			mm = imgIn[mmIdx];
			rm = imgIn[rmIdx];
			mb = imgIn[mbIdx];
			rb = imgIn[rbIdx];

		}
		else if((x == (row - 1)) && (y == 0))
		{
			lu = 0;
			mu = imgIn[muIdx];
			ru = imgIn[ruIdx];
			lm = 0;
			lb = 0;
			mm = imgIn[mmIdx];
			rm = imgIn[rmIdx];
			mb = 0;
			rb = 0;
		}
		else if(x == 0 && y == (col - 1))
		{
			lu = 0;
			mu = 0;
			ru = 0;
			lm = imgIn[lmIdx];
			lb = imgIn[lbIdx];
			mm = imgIn[mmIdx];
			rm = 0;
			mb = imgIn[mbIdx];
			rb = 0;
		}
		else if(x == (row - 1) && y == (col - 1))
		{
			lu = imgIn[luIdx];
			mu = imgIn[muIdx];
			ru = 0;
			lm = imgIn[lmIdx];
			lb = 0;
			mm = imgIn[mmIdx];
			rm = 0;
			mb = 0;
			rb = 0;
		}
		else if(x == 0 &&  y == 0)
		{
			lu = 0;
			mu = 0;
			ru = 0;
			lm = 0;
			lb = 0;
			mm = imgIn[mmIdx];
			rm = imgIn[rmIdx];
			mb = imgIn[mbIdx];
			rb = imgIn[rbIdx];
		}

		float tX, tY, T;
		tX = (-1) * lu + lb - 2 * mu + 2 * mb - ru + rb;
		tY = lu + 2 * lm + lb - ru - 2 * rm - rb;

		T = sqrt(tX * tX + tY * tY);

		if(T > 200)
			imgOut[offset] = 255;
		else if(T < 100)
			imgOut[offset] = 0;
		else
			imgOut[offset] = 0;

	}
}

void mySobel::SobelCompute(float *A, float *B)
{
	float *test = A;
	const int SIZE = row * col * sizeof(float);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float *imgIn_dev, *imgOut_dev;

	// malloc memory on GPU
	cudaState = cudaMalloc((void **)&imgIn_dev, SIZE);
	assert(cudaState == cudaSuccess);
	// copy image data from host to device
	cudaState = cudaMemcpy(imgIn_dev, A, SIZE, cudaMemcpyHostToDevice);
	assert(cudaState == cudaSuccess);
	cudaState = cudaMalloc((void **)&imgOut_dev, SIZE);
	assert(cudaState == cudaSuccess);
	cudaMemset(imgOut_dev, 0, SIZE);

	dim3 block(16, 16);
	dim3 grid((row + 15) / 16, (col + 15) / 16);
	SobelKernel<<<grid, block>>>(row, col, imgIn_dev, imgOut_dev);

	cudaState = cudaMemcpy(B, imgOut_dev, SIZE, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "GPU use time : " << elapsedTime << " ms" << endl;

	// free cuda memory
	cudaFree(imgIn_dev);
	cudaFree(imgOut_dev);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
