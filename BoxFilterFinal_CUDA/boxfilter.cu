#include "boxfilter.h"

#define INDX(r, c, w) ((r)*(w) + (c))

texture<float, cudaTextureType2D, cudaReadModeElementType> texIn;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

#define TILE_W 16
#define TILE_H 16

#define R    5   // filter radius
#define D (R * 2 + 1)        // filter diameter
#define S (D * D)            // filter size

#define BLOCK_W (TILE_W + (2 * R))
#define BLOCK_H (TILE_H + (2 * R))

inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
	if(err != cudaSuccess)
	{
		cout << file << " CUDA runtime API error at " << line << endl;
		exit(EXIT_FAILURE) ;
	}
}

BFilter::BFilter(int wid, int hei, int filterW)
{
	cudaError_t cudaState = cudaSuccess;
	cudaState = cudaMalloc((void**)&d_imgIn_, sizeof(float) * wid * hei);
	assert(cudaState == cudaSuccess);

	cudaState = cudaMalloc((void **)&d_imgOut_, sizeof(float) * wid * hei);
	assert(cudaState == cudaSuccess);

	cudaState = cudaMalloc((void **)&d_filter_, sizeof(float) * filterW * filterW);
	assert(cudaState == cudaSuccess);
}

BFilter::~BFilter()
{
	cudaFree(d_imgIn_);
	cudaFree(d_imgOut_);
	cudaFree(d_filter_);
	cudaFree(d_imgIn_Pitch_);
}

// boxfilter based on global memory
__global__ void boxfilterGlo_kernel(float *d_out, const float * d_in, int wid, int hei, const float * __restrict__ d_filter, int filterW)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if(idx >= wid || idy >= hei)
        return ;

    int filterR = (filterW - 1) / 2;
    float val = 0.f;

    for(int fr = -filterR; fr <= filterR; ++fr)
        for(int fc = -filterR; fc <= filterR; ++fc)
        {
            int ir = idy + fr;
            int ic = idx + fc;

            // check if inside image
            if((ic >= 0) && (ic < wid) && (ir >= 0) && (ir < hei))
                val += d_in[INDX(ir, ic, wid)] * d_filter[INDX(fr+filterR, fc+filterR, filterW)];
        }

    d_out[INDX(idy, idx, wid)] = val;
}

// do boxfilter on separable two dimension accumulation
// process row
__device__ void d_boxfilter_x(float *id, float *od, int w, int h, int r)
{
	float scale = 1.0f / (float)((r << 1) + 1);      // the width of filter
	float t;

	// do left edge
	t = id[0] * r;
	for(int x = 0; x < (r + 1); x++)
	{
		t += id[x];
	}

	od[0] = t * scale;

	for(int x = 1; x < (r + 1); x++)
	{
		t += id[x+r];
		t -= id[0];
		od[x] = t * scale;
	}

	// main loop
	for(int x = (r+1); x < w - r; x++)
	{
		t += id[x+r];
		t -= id[x-r-1];
		od[x] = t *scale;
	}

	// do right edge
	for(int x = w - r; x < w; x++)
	{
		t += id[w - 1];
		t -= id[x - r - 1];
		od[x] = t * scale;
	}
}
// process column
__device__ void d_boxfilter_y(float *id, float *od, int w, int h, int r)
{
	float scale = 1.0f / (float)((r << 1) + 1);

	float t;
	// do left edge
	t = id[0] * r;

	for(int y = 0; y < (r + 1); y++)
	{
		t += id[y * w];
	}

	od[0] = t * scale;

	for(int y = 1; y < (r + 1); y++)
	{
		t += id[(y+r)*w];
		t -= id[0];
		od[y * w] = t * scale;
	}

	// main loop
	for(int y = (r + 1); y < (h - r); y++)
	{
		t += id[(y + r) * w];
		t -= id[(y - r) * w - w];
		od[y * w] = t * scale;
	}

	// do right edge
	for(int y = h - r; y < h; y++)
	{
		t += id[(h - 1) * w];
		t -= id[((y - r) * w) - w];
		od[y * w] = t * scale;
	}
}

__global__ void d_boxfilter_x_global(float *id, float *od, int w, int h, int r)
{
	unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
	d_boxfilter_x(&id[y * w], &od[y*w], w, h, r);
}

__global__ void d_boxfilter_y_global(float *id, float *od, int w, int h, int r)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	d_boxfilter_y(&id[x], &od[x], w, h, r);
}

// do boxfilter on shared memory
__global__ void boxfilterSha_kernel(float *d_out, float *d_in, int wid, int hei, float *d_filter, int filterW)
{
	__shared__ float smem[BLOCK_W * BLOCK_H];
	int idx = threadIdx.x + blockDim.x * TILE_W;
	int idy = threadIdx.y + blockDim.y * TILE_H;

	idx = max(0, idx);
	idx = min(idx, wid - 1);
	idy = max(0, idy);
	idy = min(idy, hei - 1);

	unsigned int index = idy * wid + idx;         // the image index
	unsigned int bindex = threadIdx.y * blockDim.x + threadIdx.x;        // the thread index in one block

	smem[bindex] = d_in[index];
	__syncthreads();

	// only threads inside the apron will write results
	if((threadIdx.x >= R) && (threadIdx.x < (BLOCK_W - R)) && (threadIdx.y >= R) && (threadIdx.y < (BLOCK_H - R)))
	{
		float sum = 0.0f;
		for(int dy = -R; dy <= R; dy++)
			for(int dx = -R; dx <= R; dx++)
			{
				sum += smem[bindex + (dy * blockDim.x) + dx] * d_filter[INDX(dx+R, dy+R, filterW)];
			}

		d_out[index] = sum;
	}
}

// do boxfilter on texture memory
__global__ void boxfilterTex_kernel(float *d_Out, int wid, int hei, float *d_filter, int filterW)
{
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const int idy = threadIdx.y + blockDim.y * blockIdx.y;

	if(idx >= wid || idy >= hei)
		return ;

	int filterR = (filterW - 1) / 2;
	float val = 0.f;

	for(int i = -filterR; i <= filterR; ++i)      // row
	{
		for(int j = -filterR; j <= filterR; ++j)  // col
		{
			val += tex2D(texIn, idy + j, idx + i) * d_filter[INDX(i+filterR, j+filterR, filterW)];
		}
	}

	d_Out[INDX(idy, idx, wid)] = val;
}

void BFilter::boxfilterGlo(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW)
{
	cudaError_t cudaState = cudaSuccess;

	// copy data from host to device
	cudaState = cudaMemcpy(d_imgIn_, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice);
	assert(cudaState == cudaSuccess);
	cudaState = cudaMemcpy(d_filter_, filter, sizeof(float) * filterW * filterW, cudaMemcpyHostToDevice);
	assert(cudaState == cudaSuccess);

	dim3 threadPerBlock(16, 16);
	dim3 blockPerGrid;
	blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = (hei + threadPerBlock.x - 1) / threadPerBlock.y;

	boxfilterGlo_kernel<<<blockPerGrid, threadPerBlock>>>(d_imgOut_, d_imgIn_, wid, hei, d_filter_, filterW);

	cudaState = cudaMemcpy(imgOut, d_imgOut_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost);
	assert(cudaState == cudaSuccess);

	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;
}

void BFilter::boxfilterSha(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW)
{
	cudaError_t cudaState = cudaSuccess;

	// copy data from host to device
	cudaState = cudaMemcpy(d_imgIn_, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice);
	assert(cudaState == cudaSuccess);
	cudaState = cudaMemcpy(d_filter_, filter, sizeof(float) * filterW * filterW, cudaMemcpyHostToDevice);
	assert(cudaState == cudaSuccess);

	dim3 threadPerBlock(BLOCK_W, BLOCK_H);
	dim3 blockPerGrid;
	blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = (hei + threadPerBlock.x - 1) / threadPerBlock.y;

	cout << "ThreadIdx.x = " << threadPerBlock.x << endl;
	cout << "ThreadIdx.y = " << threadPerBlock.y << endl;

	boxfilterSha_kernel<<<blockPerGrid, threadPerBlock>>>(d_imgOut_, d_imgIn_, wid, hei, d_filter_, filterW);

	cudaState = cudaMemcpy(imgOut, d_imgOut_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost);
	assert(cudaState == cudaSuccess);

	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;
}

void BFilter::boxfilterSep(float *imgOut, float *imgIn, int wid, int hei, int filterW)
{
	int nthreads = 512;

	cudaError_t cudaState = cudaSuccess;
	float *d_temp;
	cudaState = cudaMalloc((void **)&d_temp, sizeof(float) * wid * hei);
	assert(cudaState == cudaSuccess);

	cudaState = cudaMemset(d_temp, 0, sizeof(float) * wid * hei);
	assert(cudaState == cudaSuccess);

	cudaState = cudaMemcpy(d_imgIn_, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice);
	assert(cudaState == cudaSuccess);

	int filterR = (filterW - 1) / 2;

	dim3 threadPerBlock;
	threadPerBlock.x = 512;
	threadPerBlock.y = 1;
	dim3 blockPerGrid;
	blockPerGrid.x = (hei + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = 1;

	// only one iteration
	d_boxfilter_x_global<<<blockPerGrid, threadPerBlock>>>(d_imgIn_, d_temp, wid, hei, filterR);

	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

	blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = 1;
	d_boxfilter_y_global<<<blockPerGrid, threadPerBlock>>>(d_temp, d_imgOut_, wid, hei, filterR);

	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(imgOut, d_imgOut_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
}

void BFilter::boxfilterTex(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW)
{
	cudaError_t cudaState = cudaSuccess;

	size_t pitch;
	cudaState = cudaMallocPitch((void **)&d_imgIn_Pitch_, &pitch, wid * sizeof(float), hei);
	assert(cudaState == cudaSuccess);

	cudaState = cudaMemcpy2D(d_imgIn_Pitch_, pitch, d_imgIn_, wid * sizeof(float), wid * sizeof(float), hei, cudaMemcpyHostToDevice);
	assert(cudaState == cudaSuccess);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	texIn.addressMode[0] = texIn.addressMode[1] = cudaAddressModeBorder;

	cudaState = cudaBindTexture2D(NULL, texIn, d_imgIn_Pitch_, channelDesc, wid, hei, pitch);
	assert(cudaState == cudaSuccess);

	dim3 threadPerBlock(16, 16);
	dim3 blockPerGrid;
	blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
	blockPerGrid.y = (hei + threadPerBlock.x - 1) / threadPerBlock.y;

	boxfilterTex_kernel<<<blockPerGrid, threadPerBlock>>>(d_imgOut_, wid, hei, d_filter_, filterW);

	cudaState = cudaMemcpy(imgOut, d_imgOut_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost);
	assert(cudaState == cudaSuccess);

	cudaUnbindTexture(texIn);

	cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;
}
