#include "boxfilter.h"

texture<float, cudaTextureType2D> texIn;

__global__ void boxfilter_kernel(float *out, int wid, int hei, const size_t pitch, const int fWid, const int fHei)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	float outputVal = 0.0f;

	if(idx < wid && idy < hei)
	{
		for(int i = -fWid; i <= fWid; ++i)
		{
			for(int j = -fHei; j <= fHei; ++j)
				outputVal += tex2D(texIn, idx + i, idy + j);
		}
		outputVal /= ((2 * fWid + 1) * (2 * fHei + 1));

		int offset = idy * pitch / sizeof(float) + idx;
		out[offset] = outputVal;
	}
}

// 2D Array Memory Version
void BFilter::boxfilter()
{
    cudaError_t cudaState = cudaSuccess;

    size_t pitch = width * sizeof(float);

    cudaChannelFormatDesc channelDescArray = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cudaArrayIn;
    cudaState = cudaMallocArray(&cudaArrayIn, &channelDescArray, width, height, cudaArrayDefault);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&dataOutD, width * height * sizeof(float));
    assert(cudaState == cudaSuccess);

    // copy data from host to cudaArray
    cudaState = cudaMemcpyToArray(cudaArrayIn, 0, 0, data, width * height * sizeof(float), cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);

    // set texture reference parameters
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    texIn.addressMode[0] = texIn.addressMode[1] = cudaAddressModeBorder;
    cudaState = cudaBindTextureToArray(texIn, cudaArrayIn, channelDesc);
    assert(cudaState == cudaSuccess);

    // launch the kernel
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (width + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (height + threadPerBlock.y - 1) / threadPerBlock.y;

    boxfilter_kernel<<<blockPerGrid, threadPerBlock>>>(dataOutD, width, height, pitch, rad, rad);

    //cudaState = cudaMemcpyToArray(data, 0, 0, cudaArrayIn, width * height * sizeof(float), cudaMemcpyDefault);
    // copy data back to host
    cudaState = cudaMemcpy(data, dataOutD, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    assert(cudaState == cudaSuccess);

    // Unbind the texture
    cudaFreeArray(cudaArrayIn);  // there is no need to call cudaUnbindTexture ... no such API
}

// 2D Linear memory version
/*
void BFilter::boxfilter()
{
	cudaError_t cudaState = cudaSuccess;

	// The pitch is the width in bytes of the allocation
	size_t pitch = 0;
	cudaState = cudaMallocPitch((void **)&dataInD, &pitch, width * sizeof(float), height);   // the requested pitched allocation width is in bytes !!!
	assert(cudaState == cudaSuccess);
	cudaState = cudaMallocPitch((void **)&dataOutD, &pitch, width * sizeof(float), height);
	assert(cudaState == cudaSuccess);

	cout << "Pitch = " << pitch << endl;

	// copy data from 2D host memory to device memory
	cudaState = cudaMemcpy2D(dataInD, pitch, data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);    // the spitch and width is also in bytes !!!
	assert(cudaState == cudaSuccess);

	// bind the dataInD to texture memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	texIn.addressMode[0] = texIn.addressMode[1] = cudaAddressModeBorder;
	cudaState = cudaBindTexture2D(0, texIn, dataInD, channelDesc,width, height, pitch);    // the width & height is in texel units, but pitch is in bytes
	assert(cudaState == cudaSuccess);

	// launch the kernel
	dim3 threadPerBlock(16, 16);
	dim3 blockPerGrid;
	blockPerGrid.x = (width + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = (height + threadPerBlock.y - 1) / threadPerBlock.y;

	boxfilter_kernel<<<blockPerGrid, threadPerBlock>>>(dataOutD, width, height, pitch, rad, rad);

	// copy data back to host
	cudaMemcpy2D(data, width * sizeof(float), dataOutD, pitch, width * sizeof(float), height, cudaMemcpyDeviceToHost);

	// Unbind the texture memory
	cudaState = cudaUnbindTexture(texIn);
	assert(cudaState == cudaSuccess);
}
*/


void BFilter::print()
{
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            if(j < width - 1)
                std::cout << data[j + i * width] << ", ";
            else
                std::cout << data[j + i * width];
        }
        std::cout << "; " << std::endl;
    }
}

BFilter::~BFilter()
{
    if(!dataInD)
        cudaFree(dataInD);
    if(!dataOutD)
        cudaFree(dataOutD);
}
