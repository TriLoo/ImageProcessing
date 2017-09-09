#include "boxfilter.h"

#define INDX(r, c, w) (((r) * (w)) + (c))

texture<float, cudaTextureType2D, cudaReadModeElementType>  texA;

// boxfilter based on texture memory
__global__ void bfilterTex(float *d_out, int wid, int hei, const float * __restrict__ d_filter, int filterW)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;    // width  == col
    int idy = threadIdx.y + blockDim.y * blockIdx.y;    // height == row

    if(idx > wid || idy > hei)
        return ;

    float val = 0.f;
    int filterRadius = (filterW - 1) / 2;

    for(int i = -filterRadius; i <= filterRadius; ++i)                    // row
        for(int j = -filterRadius; i <= filterRadius; ++j)                // col
        {
            val += tex2D(texA, idy + i,idx + j) * d_filter[INDX(i + filterRadius, j + filterRadius, filterW)];
        }

    d_out[INDX(idy, idx, wid)] = val;
}

// boxfilter based on global memory
__global__ void bfilterGlo(float *d_out, const float * d_in, int wid, int hei, const float * __restrict__ d_filter, int filterW)
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
/*
__global__ void bfilterGlo(float *d_out, const float * d_in, int wid, int hei, const float * __restrict__ d_filter, int filterW)
{
    // d_in is 1D memory on global memory
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;

    int tx = x0 + blockDim.x * blockIdx.x;        // width
    int ty = y0 + blockDim.y * blockIdx.y;        // height

    if(tx > wid || ty > hei)
        return ;

    int filterRadius = (filterW - 1) / 2;
    int extWidth = blockDim.x + filterW - 1;
    float val = 0.f;

    extern __shared__ float shareImg [];

    int fx, fy;

    // case 1 : upper left
    fx = tx - filterRadius;
    fy = ty - filterRadius;
    if(fx < 0 || fy < 0)
        shareImg[INDX(y0, x0, extWidth)] = 0;
    else
        shareImg[INDX(y0, x0, extWidth)] = d_in[INDX(ty, tx, wid)];

    // case 2 : upper right
    fx = tx + filterRadius;
    fy = ty - filterRadius;
    if(fx > wid - 1 || fy < 0)
        shareImg[INDX(y0, x0 + filterW - 1, extWidth)] = 0;
    else
        shareImg[INDX(y0, x0 + filterW - 1, extWidth)] = d_in[INDX(ty, tx, wid)];

    // case 3 : lower left
    fx = tx - filterRadius;
    fy = ty + filterRadius;
    if(fx < 0 || fy > hei - 1)
        shareImg[INDX(y0 + filterW - 1, x0, extWidth)] = 0;
    else
        shareImg[INDX(y0 + filterW - 1, x0, extWidth)] = d_in[INDX(ty, tx, wid)];

    // case 4 : lower right
    fx = tx + filterRadius;
    fy = ty + filterRadius;
    if(fx > wid - 1 || fy > hei - 1)
        shareImg[INDX(y0 + filterW - 1, x0 + filterW - 1, extWidth)] = 0;
    else
        shareImg[INDX(y0 + filterW - 1, x0 + filterW - 1, extWidth)] = d_in[INDX(ty, tx, wid)];

    // syncthreads by calling __syncthreads()
    __syncthreads();

    // do convolution
    for(int i = 0; i < filterW; ++i)     // row
        for(int j = 0; j < filterW; ++j)  // col
            val += shareImg[INDX(y0 + i, x0 + j, extWidth)] * d_filter[INDX(i, j, filterW)];

    // write result back to global memory
    // Caution : for convenient, the val is adopted as its absolute value ! ! !
    d_out[INDX(tx, ty, wid)] = val > 0 ? val : -val;
}
*/

void BFilter::boxfilter(float *d_out, const float *d_in, size_t pitch, int wid, int hei, const float *d_filter,
                        int filterW)
{
    cudaError_t cudaState = cudaSuccess;
    // d_in is 2D pitch on device
    // create 2D texture and bind it to d_in
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    texA.addressMode[0] = texA.addressMode[1] = cudaAddressModeBorder;
    //texA.filterNode = cudaFilterModePoint;
    //texA.normalized = false;

    // bind the texture memory to 2D Pitch memory
    cudaState = cudaBindTexture2D(NULL, texA, d_in, channelDesc, wid, hei, pitch);
    assert(cudaState == cudaSuccess);

    // launch the kernel
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    bfilterTex<<<blockPerGrid, threadPerBlock>>>(d_out, wid, hei, d_filter, filterW);

    // assure the kernel function is correct
    cudaState = cudaGetLastError();

    if(cudaState != cudaSuccess)
    {
        cout << "Function boxfilter basing on texture memory failed due to : ";
        cout << cudaGetErrorString(cudaState) << endl;
    }
}

// boxfilter based on global memory
void BFilter::boxfilter(float *d_out, const float *d_in, int wid, int hei, const float *d_filter, int filterW)
{
    cudaError_t cudaState = cudaSuccess;

    // prepare for the kernel launch
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    // launch the kernel function
    // Set the size of shared memory
    //size_t sharedMem_size = (threadPerBlock.x + filterW - 1) * (threadPerBlock.y + filterW - 1) * sizeof(float);
    //bfilterGlo<<<blockPerGrid, threadPerBlock, sharedMem_size>>>(d_out, d_in, wid, hei, d_filter, filterW);
    bfilterGlo<<<blockPerGrid, threadPerBlock>>>(d_out, d_in, wid, hei, d_filter, filterW);

    // assure the kernel function is correct
    cudaState = cudaGetLastError();

    if(cudaState != cudaSuccess)
    {
        cout << "Function boxfilter basing on global memory failed due to : " ;
        cout << cudaGetErrorString(cudaState) << endl;
    }
}

