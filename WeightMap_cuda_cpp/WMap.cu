#include "WMap.h"
#define INDX(r, c, w) ((r) * (w) + (c))

#define GaussW 11

WMap::WMap(int wid, int hei, int lr, int gr)
{
    int lw = 2 * lr + 1;
    int gw = 2 * gr + 1;
    // laplacian filter needs lw * lw float space
    cudaCheckErrors(cudaMalloc((void **)&d_lap_, sizeof(float) * lw * lw));
    // gaussian filter only need one row float space
    cudaCheckErrors(cudaMalloc((void **)&d_gau_, sizeof(float) * gw));
    // prepare temp memories
    cudaCheckErrors(cudaMalloc((void **)&d_tempA_, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_tempB_, sizeof(float) * wid * hei));
}

// free some dynamic device memory
WMap::~WMap()
{
    if(d_lap_)
        cudaFree(d_lap_);
    if(d_gau_)
        cudaFree(d_gau_);

    if(d_tempA_)
        cudaFree(d_tempA_);
    if(d_tempB_)
        cudaFree(d_tempB_);
}

// do absolute laplacian filter based on shared memory
__global__ void laplacianAbs_kernel(float *out, float *in, int wid, int hei, const float * __restrict__ filter, int lr)
{
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    extern __shared__ float shareMem[];

    //int TILEW = blockDim.x + 2 * lr;
    int TILEW = BLOCKSIZE + 2 * lr;

    int x, y;
    // copy data from global memory to shared memory, zero extends the border
    // case 1 : upper left
    x = idx - lr;
    y = idy - lr;
    if(x < 0 || y < 0)
        shareMem[INDX(y0, x0, TILEW)] = 0;
    else
        shareMem[INDX(y0, x0, TILEW)] = in[INDX(y, x, wid)];

    // case 2 : upper right
    x = idx + lr;
    y = idy - lr;
    if(x >= wid || y < 0)
        shareMem[INDX(y0, x0 + 2 * lr, TILEW)] = 0;
    else
        shareMem[INDX(y0, x0 + 2 * lr, TILEW)] = in[INDX(y, x, wid)];

    // case 3 : lower left
    x = idx - lr;
    y = idy + lr;
    if(x < 0 || y >= hei)
        shareMem[INDX(y0 + 2 * lr, x0, TILEW)] = 0;
    else
        shareMem[INDX(y0 + 2 * lr, x0, TILEW)] = in[INDX(y, x, wid)];

    // case 4 : lower right
    x = idx + lr;
    y = idy + lr;
    if(x >= wid || y >= hei)
        shareMem[INDX(y0 + 2 * lr, x0 + 2 * lr, TILEW)] = 0;
    else
        shareMem[INDX(y0 + 2 * lr, x0 + 2 * lr, TILEW)] = in[INDX(y, x, wid)];

    __syncthreads();

    int lw = (lr << 1) + 1;
    float val = 0.f;
    for(int i = 0; i < lw; ++i)             // row
        for(int j = 0; j < lw; ++j)         // col
            val += shareMem[INDX(y0 + i, x0 + j, TILEW)] * filter[INDX(i, j, lw)];

    // obtain the absolute value
    out[INDX(idy, idx, wid)] = val >= 0 ? val : -val;
}

// do separable gaussian filter based on CUDA
__global__ void gaussfilterRow_kernel(float *out, float *in, int wid, int hei, const float * __restrict__ filter, int gr)
{
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;

    int idx = blockDim.x * blockIdx.x + x0;
    int idy = blockDim.y * blockIdx.y + y0;

    if(idx >= wid || idy >= hei)
        return ;

    extern __shared__ float shareMem[];

    int x, y;
    int TILEW = BLOCKSIZE + 2 * gr;
    // case 1 : left apron
    x = idx - gr;
    y = idy;
    if(x < 0)
        shareMem[INDX(y0, x0, TILEW)] = 0;
    else
        shareMem[INDX(y0, x0, TILEW)] = in[INDX(y, x, wid)];

    // case 2 : right apron
    x = idx + gr;
    y = idy;
    if(x >= wid)
        shareMem[INDX(y0, x0 + 2 * gr, TILEW)] = 0;
    else
        shareMem[INDX(y0, x0 + 2 * gr, TILEW)] = in[INDX(y, x, wid)];

    __syncthreads();

    float val = 0.f;
#pragma unrool
    for(int i = 0; i < GaussW; i++)
        val += __fmul_rd(shareMem[INDX(y0, x0 + i, TILEW)], filter[i]);

    out[INDX(idy, idx, wid)] = val;
}

__global__ void gaussfilterCol_kernel(float *out, float *in, int wid, int hei, float const * __restrict__ filter, int filterR)
{
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;

    int idx = blockDim.x * blockIdx.x + x0;
    int idy = blockDim.y * blockIdx.y + y0;

    if (idx >= wid || idy >= hei)
        return;

    //__shared__ float shareMem[ * BLOCKSIZE];
    extern __shared__ float shareMem[];

    int x, y;
    // case 1 : top apron
    y = idy - filterR;
    x = idx;
    if(y < 0)
        shareMem[INDX(y0, x0, BLOCKSIZE)] = 0;
    else
        shareMem[INDX(y0, x0, BLOCKSIZE)] = in[INDX(y, x, wid)];

    // case 2 : bottom apron
    y = idy + filterR;
    x = idx;
    if(y >= hei)
        shareMem[INDX(y0 + 2 * filterR, x0, BLOCKSIZE)] = 0;
    else
        shareMem[INDX(y0 + 2 * filterR, x0, BLOCKSIZE)] = in[INDX(y, x, wid)];

    __syncthreads();

    float val = 0.f;
#pragma unroll
    for(int i = 0; i < GaussW; ++i)
        //val += shareMem[INDX(y0 + i, x0, BLOCKSIZE)] * filter[i];
        val += __fmul_rd(shareMem[INDX(y0+i, x0, BLOCKSIZE)], filter[i]);

    out[INDX(idy, idx, wid)] = val;
}

// comparasion kernel
__global__ void comparison_kernel(float *outA, float *outB, float *inA, float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = idx + idy * wid;

    if(idx >= wid || idy >= hei)
        return ;

    int val = (inA[offset] >= inB[offset]) ? 1 : 0;
    outA[offset] = val;
    outB[offset] = 1 - val;
}

// Caution
// Error : the '[]' operator of vector is host function and cannot run with __global__ configuration
/*
__global__ void comparison_kernel(vector<float *> &out, vector<float *> &in, int wid, int hei)
{
    int val = in[0][offset] >= in[1][offset] ? 1 : 0;
    out[0][offset] = val;
    out[1][offset] = 1 - val;
}
*/
// laplacian filter
void WMap::laplacianAbs(float *d_imgOut, float *d_imgIn, int wid, int hei, int lr)
{
    int lw = lr * 2 + 1;
    float *lapfilter = new float [lw * lw];

    cudaStream_t st;
    cudaCheckErrors(cudaStreamCreate(&st));

    // 3 * 3 laplacian filter
    lapfilter[0] = -1;      lapfilter[1] = -1;      lapfilter[2] = -1;
    lapfilter[3] = -1;      lapfilter[4] = 8;      lapfilter[5] = -1;
    lapfilter[6] = -1;      lapfilter[7] = -1;      lapfilter[8] = -1;

    // copy laplacian filter from host to device
    cudaCheckErrors(cudaMemcpy(d_lap_, lapfilter, sizeof(float) * lw * lw, cudaMemcpyHostToDevice));

    // do kernel on GPU
    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    // launch kernel function
    int TileW = BLOCKSIZE + 2 * lr;
    laplacianAbs_kernel<<<blockPerGrid, threadPerBlock, sizeof(float) * TileW * TileW, st>>>(d_imgOut, d_imgIn, wid, hei, d_lap_, lr);

    cout << "Laplacian filter : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    if(lapfilter)
        delete [] lapfilter;

    cudaCheckErrors(cudaStreamDestroy(st));
}

void WMap::laplacianAbsTest(float *imgOut, float *imgIn, int wid, int hei, int lr)
{
    float *d_imgIn, *d_imgOut;

    cudaCheckErrors(cudaMalloc((void **)&d_imgIn, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemcpy(d_imgIn, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));

    cudaCheckErrors(cudaMalloc((void **)&d_imgOut, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemset(d_imgOut, 0, sizeof(float) * wid * hei));

    laplacianAbs(d_imgOut, d_imgIn, wid, hei, lr);

    cudaCheckErrors(cudaMemcpy(imgOut, d_imgOut, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    cudaFree(d_imgIn);
    cudaFree(d_imgOut);
}

void WMap::gaussian(float *d_imgOut, float *d_imgIn, int wid, int hei, int gr, int gsigma)
{
    int filterW = 2 * gr + 1;
    float *filter = new float [filterW];

    cudaStream_t st;
    cudaCheckErrors(cudaStreamCreate(&st));

    // generate row filter :
    filter[0] = -0.0663;    filter[1] = -0.0794;    filter[2] = -0.0914;
    filter[3] = -0.1010;    filter[4] = -0.1072;    filter[5] = -0.1094;
    filter[6] = -0.1072;    filter[7] = -0.1010;    filter[8] = -0.0914;
    filter[9] = -0.0794;    filter[10] = -0.0663;


    // copy filter data from host to device
    cudaCheckErrors(cudaMemcpy(d_gau_, filter, sizeof(float) * filterW, cudaMemcpyHostToDevice));

    // prepare needed memory on device
    float *d_temp;
    cudaCheckErrors(cudaMalloc((void **)&d_temp, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemset(d_temp, 0, sizeof(float) * wid * hei));

    int TileW = BLOCKSIZE + 2 * gr;

    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    gaussfilterRow_kernel<<<blockPerGrid, threadPerBlock, sizeof(float) * TileW * BLOCKSIZE, st>>>(d_temp, d_imgIn, wid, hei, d_gau_, gr);
    cout << "In gaussian filter ROW part : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    gaussfilterCol_kernel<<<blockPerGrid, threadPerBlock, sizeof(float) * TileW * BLOCKSIZE, st>>>(d_imgOut, d_temp, wid, hei, d_gau_, gr);
    cout << "In gaussian filter COL part : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cudaStreamDestroy(st);

    if(d_temp)
        cudaFree(d_temp);
    if(filter)
        delete [] filter;
}

void WMap::gaussianTest(float *imgOut, float *imgIn, int wid, int hei, int gr, int gsigma)
{
    float *d_imgIn, *d_imgOut;

    cudaCheckErrors(cudaMalloc((void **)&d_imgIn, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemcpy(d_imgIn, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));

    cudaCheckErrors(cudaMalloc((void **)&d_imgOut, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemset(d_imgOut, 0, sizeof(float) * wid * hei));

    gaussian(d_imgOut, d_imgIn, wid, hei, gr, 0.1);

    cudaCheckErrors(cudaMemcpy(imgOut, d_imgOut, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    cudaFree(d_imgIn);
    cudaFree(d_imgOut);
}

// do the saliency map generation
void WMap::saliencymapTest(float *imgOut, float *imgIn, int wid, int hei, int lr, int gr, double gsigma)
{
    float *d_imgIn, *d_imgOut;

    cudaCheckErrors(cudaMalloc((void **)&d_imgIn, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemcpy(d_imgIn, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));

    cudaCheckErrors(cudaMalloc((void **)&d_imgOut, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMemset(d_imgOut, 0, sizeof(float) * wid * hei));

    laplacianAbs(d_imgOut, d_imgIn, wid, hei, lr);
    gaussian(d_imgOut, d_imgOut, wid, hei, gr, gsigma);

    cudaCheckErrors(cudaMemcpy(imgOut, d_imgOut, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    cudaFree(d_imgIn);
    cudaFree(d_imgOut);
}

// do the weighted map
/*
void WMap::weightedmap(float *d_imgOut, float *d_imgIn, int wid, int hei, int lr, int gr, int gsigma, int guir,
                       double eps)
{
}
 */

void WMap::weightedmap(float *d_imgOutA, float *d_imgOutB, float *d_imgInA, float *d_imgInB, int wid, int hei, int lr,
                       int gr, int gsigma, int guir, double eps)
{
    laplacianAbs(d_tempA_, d_imgInA, wid, hei, lr);
    laplacianAbs(d_tempB_, d_imgInB, wid, hei, lr);

    //gaussian(d_imgOutA, d_tempA_, wid, hei, gr, gsigma);
    //gaussian(d_imgOutB, d_tempB_, wid, hei, gr, gsigma);
    gaussian(d_tempA_, d_tempA_, wid, hei, gr, gsigma);
    gaussian(d_tempB_, d_tempB_, wid, hei, gr, gsigma);

    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;
    comparison_kernel<<<blockPerGrid, threadPerBlock>>>(d_tempA_, d_tempB_, d_tempA_, d_tempB_, wid, hei);

    GFilter gf(wid, hei);
    gf.guidedfilter(d_imgOutA, d_imgInA, d_tempA_, wid, hei, guir, eps);
    gf.guidedfilter(d_imgOutB, d_imgInB, d_tempB_, wid, hei, guir, eps);
}

void WMap::weightedmapTest(float *imgOutA, float *imgOutB, float *imgInA, float *imgInB, int wid, int hei, int lr,
                           int gr, int gsigma, int guir, double eps)
{
    cudaEvent_t cudaStart, cudaStop;

    float *d_imgInA, *d_imgOutA, *d_imgInB, *d_imgOutB;
    cudaCheckErrors(cudaMalloc((void **)&d_imgInA, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgInB, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgOutA, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgOutB, sizeof(float) * wid * hei));

    cudaCheckErrors(cudaMemcpy(d_imgInA, imgInA, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_imgInB, imgInB, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));

    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);

    cudaEventRecord(cudaStart, 0);

    weightedmap(d_imgOutA, d_imgOutB, d_imgInA, d_imgInB, wid, hei, lr, gr, gsigma, guir, eps);

    cudaEventRecord(cudaStop, 0);
    cudaEventSynchronize(cudaStop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, cudaStart, cudaStop);
    cout << "Weighted Map GPU Time (no memory copy) : " << elapsedTime << " ms" << endl;

    cudaCheckErrors(cudaMemcpy(imgOutA, d_imgOutA, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
    cudaCheckErrors(cudaMemcpy(imgOutB, d_imgOutB, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
}

/*
void WMap::weightedmapTest(float *imgOutA, float *imgOutB, float *imgInA, float *imgInB, int wid, int hei, int lr,
                           int gr, int gsigma, int guir, double eps)
{
    cudaEvent_t cudaStart, cudaStop;

    float *d_imgInA, *d_imgOutA, *d_imgInB, *d_imgOutB;
    cudaCheckErrors(cudaMalloc((void **)&d_imgInA, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgInB, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgOutA, sizeof(float) * wid * hei));
    cudaCheckErrors(cudaMalloc((void **)&d_imgOutB, sizeof(float) * wid * hei));

    cudaCheckErrors(cudaMemcpy(d_imgInA, imgInA, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_imgInB, imgInB, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));

    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);

    cudaEventRecord(cudaStart, 0);
    laplacianAbs(d_tempA_, d_imgInA, wid, hei, lr);
    laplacianAbs(d_tempB_, d_imgInB, wid, hei, lr);

    //gaussian(d_imgOutA, d_tempA_, wid, hei, gr, gsigma);
    //gaussian(d_imgOutB, d_tempB_, wid, hei, gr, gsigma);
    gaussian(d_tempA_, d_tempA_, wid, hei, gr, gsigma);
    gaussian(d_tempB_, d_tempB_, wid, hei, gr, gsigma);

    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;
    comparison_kernel<<<blockPerGrid, threadPerBlock>>>(d_tempA_, d_tempB_, d_tempA_, d_tempB_, wid, hei);

    GFilter gf(wid, hei);
    gf.guidedfilter(d_imgOutA, d_imgInA, d_tempA_, wid, hei, guir, eps);
    gf.guidedfilter(d_imgOutB, d_imgInB, d_tempB_, wid, hei, guir, eps);

    cudaEventRecord(cudaStop, 0);
    cudaEventSynchronize(cudaStop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, cudaStart, cudaStop);
    cout << "Weighted Map GPU Time (no memory copy) : " << elapsedTime << " ms" << endl;
    */
    /*
    // now, d_temps store the saliency maps
    gaussian(d_tempA_, d_tempA_, wid, hei, gr, gsigma);
    gaussian(d_tempB_, d_tempB_, wid, hei, gr, gsigma);

    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;
    comparison_kernel<<<blockPerGrid, threadPerBlock>>>(d_tempA_, d_tempB_, d_tempA_, d_tempB_, wid, hei);

    cout << "Comparison kernel : " << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    GFilter gf(wid, hei);
    gf.guidedfilter(d_imgOutA, d_imgInA, d_tempA_, wid, hei, guir, eps);
    gf.guidedfilter(d_imgOutB, d_imgInB, d_tempB_, wid, hei, guir, eps);
    */

/*
    cudaCheckErrors(cudaMemcpy(imgOutA, d_imgOutA, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
    cudaCheckErrors(cudaMemcpy(imgOutB, d_imgOutB, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
    //cudaCheckErrors(cudaMemcpy(imgOutA, d_tempA_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
    //cudaCheckErrors(cudaMemcpy(imgOutB, d_tempB_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
}
*/
