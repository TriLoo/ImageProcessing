#include "gaussfilter.h"

#define cudaErrorCheck(err) __checkCUDAError(err, __FILE__, __LINE__)

#define BLOCKSIZE 16
#define FILTERRAD 5
#define TILE_WIDTH (BLOCKSIZE + 2 * FILTERRAD)

#define INDX(r, c, w) ((r) * (w) + (c))

#define FILTERSIZE_ (11*11)

__constant__ float d_filter_const_[FILTERSIZE_];

// declare the texture memory
texture<float, cudaTextureType2D, cudaReadModeElementType> texIn;

inline void __checkCUDAError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        cout << err << " in " << file << " at " << line << " line.";
        exit(EXIT_FAILURE);
    }
}

// filter is on constant memory
__global__ void
gaussfilterCon_kernel(float *d_imgOut, float *d_imgIn, int wid, int hei, const float *__restrict__ d_filter,
                      int filterW) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx > wid || idy > hei)
        return;

    int filterR = (filterW - 1) / 2;

    float val = 0.f;

    for (int fr = -filterR; fr <= filterR; ++fr)           // row
        for (int fc = -filterR; fc <= filterR; ++fc)      // col
        {
            int ir = idy + fr;
            int ic = idx + fc;

            if ((ic >= 0) && (ic <= wid - 1) && (ir >= 0) && (ir <= hei - 1))
                val += d_imgIn[INDX(ir, ic, wid)] * d_filter_const_[INDX(fr + filterR, fc + filterR, filterW)];
        }
    d_imgOut[INDX(idy, idx, wid)] = val;

}

__global__ void
gaussfilterGlo_kernel(float *d_imgOut, float *d_imgIn, int wid, int hei, const float *__restrict__ d_filter,
                      int filterW) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx > wid || idy > hei)
        return;

    int filterR = (filterW - 1) / 2;

    float val = 0.f;

    for (int fr = -filterR; fr <= filterR; ++fr)           // row
        for (int fc = -filterR; fc <= filterR; ++fc)      // col
        {
            int ir = idy + fr;
            int ic = idx + fc;

            if ((ic >= 0) && (ic <= wid - 1) && (ir >= 0) && (ir <= hei - 1))
                val += d_imgIn[INDX(ir, ic, wid)] * d_filter[INDX(fr + filterR, fc + filterR, filterW)];
        }
    d_imgOut[INDX(idy, idx, wid)] = val;
}

__global__ void gaussfilterTex_kernel(float *d_imgOut, float *d_imgIn, int wid, int hei, float *d_filter, int filterW) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx > wid || idy > hei)
        return;

    int filterR = (filterW - 1) / 2;

    float val = 0.f;

    for (int fr = -filterR; fr <= filterR; fr++)
        for (int fc = -filterR; fc <= filterR; fc++) {
            val += tex2D(texIn, idy + fr, idx + fc) * d_filter[INDX(fr + filterR, fc + filterR, filterW)];
        }

    d_imgOut[INDX(idy, idx, wid)] = val;
}

/*
__global__ void gaussfilterSha_kernel(float *d_imgOut, float *d_imgIn, int wid, int hei, float *d_filter, int filterW)
{
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;

    int idx = x0 + blockDim.x * blockIdx.x;
    int idy = y0 + blockDim.y * blockIdx.y;

    if(idx >= wid || idy >= hei)
        return ;

    __shared__ float shareMem[TILE_WIDTH * TILE_WIDTH];

    int x, y;

    // case 1 : upper left
    x = idx - FILTERRAD;
    y = idx - FILTERRAD;
    if(x < 0 || y < 0)
        shareMem[INDX(y0, x0, TILE_WIDTH)] = 0;
    else
        shareMem[INDX(y0, x0, TILE_WIDTH)] = d_imgIn[INDX(y, x, wid)];
    // shareMem[INDX(y0, x0, TILE_WIDTH)] = d_imgIn[INDX(idy, idx, wid) - FILTERRAD - FILTERRAD * wid]

    // case 2 : upper right
    x = idx + FILTERRAD;
    y = idy - FILTERRAD;
    if(x >= wid || y < 0)
        shareMem[INDX(y0, x0+2*FILTERRAD, TILE_WIDTH)] = 0;
    else
        shareMem[INDX(y0, x0+2*FILTERRAD, TILE_WIDTH)] = d_imgIn[INDX(y, x, wid)];
    //  shareMem[INDX(y0, x0+FILTERRAD, TILE_WIDTH)] = d_imgIn[INDX(idy, idx, wid) + FILTERRAD - FILTERRAD * wid]

    // case 3 : lower left
    x = idx - FILTERRAD;
    y = idy + FILTERRAD;
    if(x < 0 || y >= hei)
        shareMem[INDX(y0+2*FILTERRAD, x0, TILE_WIDTH)] = 0;
    else
        shareMem[INDX(y0+2*FILTERRAD, x0, TILE_WIDTH)] = d_imgIn[INDX(y, x, wid)];
    //  shareMem[INDX(y0+FILTERRAD, x0, TILE_WDITH)] = d_imgIn[INDX(idy, idx, wid) - FILTERRAD + FILTERRAD * wid]

    // case 4 : lower right
    x = idx + FILTERRAD;
    y = idy + FILTERRAD;
    if(x >= wid || y >= hei)
        shareMem[INDX(y0+2*FILTERRAD, x0+2*FILTERRAD, TILE_WIDTH)] = 0;
    else
        shareMem[INDX(y0+2*FILTERRAD, x0+2*FILTERRAD, TILE_WIDTH)] = d_imgIn[INDX(y, x, wid)];
    //  shareMem[INDX(y0+FILTERRAD, x0+FILTERRAD, TILE_WIDTH)] = d_imgIn[INDX(idy, idx, wid) + FILTERRAD + FILTERRAD * wid]

    __syncthreads();

    float val = 0.f;

    for(int fr = 0; fr <= filterW; fr++)
        for(int fc = 0; fc <= filterW; fc++)
        {
            val += shareMem[INDX(y0+fr, x0 + fc, TILE_WIDTH)] * d_filter[INDX(fr, fc, filterW)];
        }

    //assert(val > 0);
    d_imgOut[INDX(idy, idx, wid)] = val;
}
*/

__global__ void gaussfilterSha_kernel(float *d_imgOut, float *d_imgIn, int wid, int hei, float *d_filter, int filterW) {
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;

    int idx = blockDim.x * blockIdx.x + x0;
    int idy = blockDim.y * blockIdx.y + y0;

    if (idx >= wid || idy >= hei)
        return;

    int filterR = (filterW - 1) / 2;

    __shared__ float shareMem[TILE_WIDTH * TILE_WIDTH];

    int x, y;

    // case 1 : upper left
    x = idx - FILTERRAD;
    y = idy - FILTERRAD;
    if (x < 0 || y < 0)
        shareMem[INDX(y0, x0, TILE_WIDTH)] = 0;
    else
        shareMem[INDX(y0, x0, TILE_WIDTH)] = d_imgIn[INDX(idy, idx, wid) - FILTERRAD - INDX(FILTERRAD, 0, wid)];

    // case 2 : upper right
    x = idx + FILTERRAD;
    y = idy - FILTERRAD;
    if (x >= wid || y < 0)
        shareMem[INDX(y0, x0 + 2*FILTERRAD, TILE_WIDTH)] = 0;
    else
        shareMem[INDX(y0, x0 + 2*FILTERRAD, TILE_WIDTH)] = d_imgIn[INDX(idy, idx, wid) + FILTERRAD -
                                                               INDX(FILTERRAD, 0, wid)];

    // case 3 : lower left
    x = idx - filterR;
    y = idy + filterR;
    if (x < 0 || y >= hei)
        shareMem[INDX(y0 + 2*filterR, x0, TILE_WIDTH)] = 0;
    else
        shareMem[INDX(y0 + 2*filterR, x0, TILE_WIDTH)] = d_imgIn[INDX(idy, idx, wid) - FILTERRAD +
                                                               INDX(FILTERRAD, 0, wid)];

    // case 4 : lower right
    x = idx + filterR;
    y = idy + filterR;
    if (x >= wid || y >= hei)
        shareMem[INDX(y0 + 2*FILTERRAD, x0 + 2*FILTERRAD, TILE_WIDTH)] = 0;
    else
        shareMem[INDX(y0 + 2*FILTERRAD, x0 + 2*FILTERRAD, TILE_WIDTH)] = d_imgIn[INDX(idy, idx, wid) + FILTERRAD + INDX(FILTERRAD, 0, wid)];

    __syncthreads();

    // convolution
    float sum = 0.f;

    x = FILTERRAD + threadIdx.x;
    y = FILTERRAD + threadIdx.y;
    for (int i = -FILTERRAD; i <= FILTERRAD; ++i)                       // row
        for (int j = -FILTERRAD; j <= FILTERRAD; ++j)                   // col
            sum += shareMem[INDX(y + j, x + i, TILE_WIDTH)] * d_filter[INDX(j + FILTERRAD, i + FILTERRAD, filterW)];

    d_imgOut[INDX(idy, idx, wid)] = sum;
}

GFilter::GFilter(int wid, int hei, int filterW, float sig) {
    cudaError_t cudaState = cudaSuccess;
    cudaState = cudaMalloc((void **) &d_imgIn_, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **) &d_imgOut_, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **) &d_filter_, sizeof(float) * filterW * filterW);
    assert(cudaState == cudaSuccess);

    filterW_ = filterW;
    filterR_ = (filterW - 1) / 2;
    filterSize_ = filterW * filterW;
    sig_ = sig;

    filter_ = new float[filterW * filterW];
}

GFilter::~GFilter() {
    if (filter_)
        delete[] filter_;
    if (d_filter_)
        cudaFree(d_filter_);
    if (d_imgOut_)
        cudaFree(d_imgOut_);
    if (d_imgIn_)
        cudaFree(d_imgIn_);
}

// prepare the gaussian filter
void GFilter::createfilter() {
    //cudaError_t cudaState = cudaSuccess;

    float val = 0.f;
    float sum = 0.f;

    float sig = 2 * sig_ * sig_;
    for (int i = -filterR_; i <= filterR_; ++i)       // row
    {
        for (int j = -filterR_; j <= filterR_; ++j)   // col
        {
            val = i * i + j * j;
            val = exp(-val / sig) / (sig * PI);
            sum += val;
            int offset = (i + filterR_) * filterW_ + j + filterR_;
            filter_[offset] = val;
        }
    }

    for (int i = 0; i < filterSize_; i++)
        filter_[i] *= 1.0 / sum;

    cudaErrorCheck(cudaMemcpy(d_filter_, filter_, sizeof(float) * filterSize_, cudaMemcpyHostToDevice));

    // copy data from host to constant memory on device
    cudaErrorCheck(
            cudaMemcpyToSymbol(d_filter_const_, filter_, sizeof(float) * filterSize_, NULL, cudaMemcpyHostToDevice));
}

// copy data from host to device including filter & image data
void GFilter::prepareMemory(float *imgIn, int wid, int hei) {
    //cudaError_t cudaState = cudaSuccess;
    cudaErrorCheck(cudaMemcpy(d_imgIn_, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice));
}

// do gaussian filtering on global memory
void GFilter::gaussfilterGlo(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW) {

    createfilter();
    prepareMemory(imgIn, wid, hei);

    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;

    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    gaussfilterGlo_kernel << < blockPerGrid, threadPerBlock >> > (d_imgOut_, d_imgIn_, wid, hei, d_filter_, filterW_);
    //gaussfilterCon_kernel<<<blockPerGrid, threadPerBlock>>>(d_imgOut_, d_imgIn_, wid, hei, d_filter_, filterW_);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "GPU Time = :" << elapsedTime << " ms" << endl;

    cudaErrorCheck(cudaMemcpy(imgOut, d_imgOut_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));

    cout << "In gaussfilterGlo Function :" << cudaGetErrorString(cudaPeekAtLastError()) << endl;
}

void GFilter::gaussfilterTex(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW) {
    createfilter();

    size_t pitch;

    float *d_imgIn_Pitch;
    cudaErrorCheck(cudaMallocPitch((void **) &d_imgIn_Pitch, &pitch, wid * sizeof(float), hei));

    // copy image data from host to 2D Pitch
    cudaErrorCheck(cudaMemcpy2D(d_imgIn_Pitch, pitch, imgIn, wid * sizeof(float), wid * sizeof(float), hei,
                                cudaMemcpyHostToDevice));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    texIn.addressMode[0] = texIn.addressMode[1] = cudaAddressModeBorder;
    //texIn.addressMode[0] = cudaAddressModeBorder;
    //texIn.addressMode[1] = cudaAddressModeBorder;
    // bind the texture to 2D Pitch
    cudaErrorCheck(cudaBindTexture2D(NULL, texIn, d_imgIn_Pitch, channelDesc, wid, hei, pitch));

    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;

    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    gaussfilterTex_kernel <<< blockPerGrid, threadPerBlock >>>
                                             (d_imgOut_, d_imgIn_Pitch, wid, hei, d_filter_, filterW_);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "GPU Time = :" << elapsedTime << " ms" << endl;

    cudaUnbindTexture(texIn);

    cudaErrorCheck(cudaMemcpy(imgOut, d_imgOut_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
}

void GFilter::gaussfilterSha(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW) {
    createfilter();
    prepareMemory(imgIn, wid, hei);


    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    gaussfilterSha_kernel<<< blockPerGrid, threadPerBlock>>>(d_imgOut_, d_imgIn_, wid, hei, d_filter_, filterW);

    cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "GPU Time = :" << elapsedTime << " ms" << endl;

    cudaErrorCheck(cudaMemcpy(imgOut, d_imgOut_, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost));
}

void GFilter::gaussfilterSep(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW)
{
}

void GFilter::gaussfilterShaSep(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW)
{

}
