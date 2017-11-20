#include "BFilter.h"

BFilter::~BFilter()
{

}

// do boxfilter on separable two dimension accumulation
// process row
// 加运算比移位运算优先级高
__device__ void d_boxfilter_x(float *Out, float *imgIn, int wid, int hei, int filterR)
{
	float scale = 1.0f / (float)((filterR << 1) + 1);
	//float scale = 0.0322581;
	float t;

	// do the left edge
	t = imgIn[0] * filterR;
	for(int x = 0; x < (filterR + 1); x++)
	{
		t += imgIn[x];
	}

	Out[0] = t * scale;

	for(int x = 1; x < (filterR + 1); x++)
	{
		t += imgIn[x + filterR];
		t -= imgIn[0];
		Out[x] = t * scale;
	}

	// main loop
	for(int x = (filterR + 1); x < (wid - filterR); x++)
	{
		t += imgIn[x + filterR];
		t -= imgIn[x - filterR - 1];
		Out[x] = t * scale;
	}

	// do the right edge
	for(int x = (wid - filterR); x < wid; x++)
	{
		t += imgIn[wid - 1];
		t -= imgIn[x - filterR - 1];
		Out[x] = t *  scale;
	}
}

// process column
__device__ void d_boxfilter_y(float *imgOut,float *imgIn, int wid, int hei, int filterR)
{
	float scale = 1.0f / (float)((filterR << 1) + 1);
	//float scale = 0.0322581;

	float t;

	// do the upper edge
	t = imgIn[0] * filterR;
	for(int y = 0; y < (filterR + 1); y++)
	{
		t += imgIn[y * wid];
	}

	imgOut[0] = 1.0 * t * scale;

	for(int y = 1; y < (filterR + 1); y++)
	{
		t += imgIn[(y + filterR) * wid];
		t -= imgIn[0];
		imgOut[y * wid] = t * scale;
	}

	// main loop
	for(int y = filterR + 1; y < hei - filterR; y++)
	{
		t += imgIn[(y + filterR) * wid];
		t -= imgIn[(y - filterR - 1) * wid];
		imgOut[y * wid] = t * scale;
	}

	// do the bottom dege
	for(int y = hei - filterR; y < hei; y++)
	{
		t += imgIn[(hei - 1) * wid];
		t -= imgIn[(y - filterR - 1) * wid];
		imgOut[y * wid] = t * scale;
	}
}

__global__ void d_boxfilter_x_global(float *Out, float *In, int wid, int hei, int filterR)
{
	unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
	if( y >= hei)
		return ;
	d_boxfilter_x(&Out[y * wid], &In[y * wid], wid, hei, filterR);
}

__global__ void d_boxfilter_y_global(float *Out, float *In, int wid, int hei, int filterR)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= wid)
		return ;

	d_boxfilter_y(&Out[x], &In[x], wid, hei, filterR);
}

void BFilter::boxfilter(float *d_imgOut, float *d_imgIn, int wid, int hei, int filterR)
{
    int nthreads = 512;

    float *d_temp;
    cudaCheckErrors(cudaMalloc((void **)&d_temp, sizeof(float) * wid * hei));

    cudaCheckErrors(cudaMemset(d_temp, 0, sizeof(float) * wid * hei));

    dim3 threadPerBlock(nthreads, 1);
    dim3 blockPerGrid;
    blockPerGrid.x = (hei + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = 1;

    // only one iteration
    d_boxfilter_x_global<<<blockPerGrid, threadPerBlock>>>(d_temp, d_imgIn, wid, hei, filterR);
    //cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = 1;
    d_boxfilter_y_global<<<blockPerGrid, threadPerBlock>>>(d_imgOut, d_temp, wid, hei, filterR);
    //cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    //cudaCheckErrors(cudaDeviceSynchronize());

    if(d_temp)
        cudaFree(d_temp);
}

void BFilter::boxfilterTest(float *imgOut, float *imgIn, int wid, int hei, int filterR)
{
	int nthreads = 512;

	float *d_temp;
	cudaCheckErrors(cudaMalloc((void **)&d_temp, sizeof(float) * wid * hei));
	cudaCheckErrors(cudaMemset(d_temp, 0, sizeof(float) * wid * hei));

	float *d_imgIn, *d_imgOut;
	cudaCheckErrors(cudaMalloc((void **)&d_imgIn, sizeof(float) * wid * hei));
	cudaCheckErrors(cudaMemcpy(d_imgIn, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));
	cudaCheckErrors(cudaMalloc((void **)&d_imgOut, sizeof(float) * wid * hei));

    boxfilter(d_imgOut, d_imgIn, wid, hei, filterR);

	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

	cudaCheckErrors(cudaMemcpy(imgOut, d_imgOut, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
	cudaCheckErrors(cudaDeviceSynchronize());

	if(d_temp)
		cudaFree(d_temp);
	if(d_imgIn)
		cudaFree(d_imgIn);
	if(d_imgOut)
		cudaFree(d_imgOut);
}