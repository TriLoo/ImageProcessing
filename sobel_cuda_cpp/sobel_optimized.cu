#include "sobel.h"

// declare texutre reference globally
texture<float, cudaTextureType2D, cudaReadModeElementType> texIn;

__constant__ float *sobelX;
__constant__ float *sobelY;

__global__ void SobelKernel(int row, int col, float *imgOut )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	//int index = x + y * blockDim.x * gridDim.x;
	int index = x + y * col;

	if(x < col && y < row)
	{
		// do the calculation parallelized
		float lu = tex2D(texIn, x - 1, y - 1);
	    float lm = tex2D(texIn, x, y - 1);
		float lb = tex2D(texIn, x + 1, y - 1);

		float mu = tex2D(texIn, x - 1, y);
		float mm = tex2D(texIn, x, y);
		float mb = tex2D(texIn, x + 1, y);

		float ru = tex2D(texIn, x - 1, y + 1);
		float rm = tex2D(texIn, x , y +1);
		float rb = tex2D(texIn, x + 1, y + 1);

		float tX = 0, tY = 0, T = 0;
		tX = (-1) * lu + lb - 2 * mu + 2 * mb - ru + rb;
		tY = lu + 2 * lm + lb - ru - 2 * rm - rb;

		T = sqrt(tX * tX + tY * tY);

		// Now the 200 is the threshold value
		if(T > 100)
			imgOut[index] = 255;
		else if(T < 50)
			imgOut[index] = 0;
		else
			imgOut[index] = T;
	}
}

__global__ void copy_texture_kernel(float *iptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockIdx.x * blockDim.x;

	float c = tex2D(texIn, x, y);
	if(c != 0)
		iptr[offset] = c;
}

// float *A is the image data stored on host
void mySobel::SobelCompute(float *A, float *B)
{
	cudaError_t cudaState_1 = cudaSuccess;

	cout << "row = " << row << "col = " << col << endl;

	// declare sobel operator matrix
	float temp_sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	float temp_sobelY[9] = {-1, -1, -1, 0, 0, 0, 1, 2, 1};
	cout << "sizeof(*temp_sobleX) : " << sizeof(*temp_sobelX) << endl;
	cudaMemcpyToSymbol(sobelX, temp_sobelX, sizeof(float) * 9);  // no need to malloc memory for constant variables
	cudaMemcpyToSymbol(sobelY, temp_sobelY, sizeof(float) * 9);


	// measure the performance with the help of envent
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *imgIn_dev, *imgOut_dev;

	const int SIZE = row * col * sizeof(float);

	cudaState = cudaMalloc((void**)&imgIn_dev, SIZE);
	assert(cudaState == cudaSuccess);
	cudaState = cudaMalloc((void **)&imgOut_dev, SIZE);
	assert(cudaState == cudaSuccess);
	cudaMemset(imgOut_dev, 0, SIZE);

	// copy the image data from host to device
	cudaState_1 = cudaMemcpy(imgIn_dev, A, SIZE, cudaMemcpyHostToDevice);
	assert(cudaState_1 == cudaSuccess);

	// bind the cuda memory to the texture memory

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	// create 2D array
	cudaArray* imgArray;
	cudaMallocArray(&imgArray, &desc, col, row);
	// copy  to device memory some data locate at input image
	cudaMemcpyToArray(imgArray, 0, 0, A, SIZE, cudaMemcpyHostToDevice);
	// set texture parameters;
	texIn.addressMode[0] = cudaAddressModeWrap;
	texIn.addressMode[1] = cudaAddressModeWrap;
	texIn.filterMode = cudaFilterModeLinear;
	texIn.normalized = false;
	cudaBindTextureToArray(texIn, imgArray, desc);
	//cudaBindTexture2D(NULL, texIn, imgArray, desc, col, row, sizeof(float) * col);
	//	cudaBindTexture(NULL, texOut, imgOut_dev, SIZE);

	// Call the kernel function
	dim3 thread(16, 16);    // 16 * 16 threads per block
	dim3 grid((row + 15)/16, (col + 15)/ 16);

	cudaEventRecord(start, 0);
	SobelKernel<<<grid, thread>>>(row, col, imgOut_dev);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Sobel calculated on GPU cost " << elapsedTime << " ms" << endl;

	// copy result back to host
	cudaMemcpy(B, imgOut_dev, SIZE, cudaMemcpyDeviceToHost);
	// for test
	for(int i = 0; i < 10; i++)
	{
		cout << B[i] << endl;
	}

	// clean up memory allocated on The GPU
	cudaUnbindTexture(texIn);
	//cudaUnbindTexture(TexOut);
	cudaFreeArray(imgArray);
	cudaFree(imgIn_dev);
	cudaFree(imgOut_dev);
	// destroy the event variables
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
