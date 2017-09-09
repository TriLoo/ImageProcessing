#include "weightmap.h"

const float PI = 3.14159;

__global__ void compare_kernel(float *outA, float *outB, const float *inA, const float *inB, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx > wid || idy > hei)
        return ;

    const int offset =  idy * wid + idx;
    float valA = inA[offset];
    float valB = inB[offset];

    outA[offset] = valA > valB ? 1 : 0;
    outB[offset] = valA > valB ? 0 : 1;
}


// create laplace & gauss filters and copy them from host to device
void WMap::createfilterLG()
{
    cudaError_t cudaState = cudaSuccess;

    int lap_wid = 2 * lap_rad + 1;
    int lap_size = lap_wid * lap_wid;
    int gau_wid = 2 * gau_rad + 1;
    int gau_size = gau_wid * gau_wid;
    //float *laplafilter, *guassfilter;
    float *lap_filter = new float [lap_size];
    float *gau_filter = new float [gau_size];

    cout << "Lap size = " << lap_size << endl;
    cout << "Gau size = " << gau_size << endl;

    // make lap_rad = 1
    // generate the 3 * 3 laplacian filter
    // for more details can see : "数字图像处理与机器视觉" 一书
    lap_filter[0] = -1;     lap_filter[1] = -1;     lap_filter[2] = -1;
    lap_filter[3] = -1;     lap_filter[4] = 8;      lap_filter[5] = -1;
    lap_filter[6] = -1;     lap_filter[7] = -1;     lap_filter[8] = -1;

    cudaState = cudaMalloc((void **)&d_lap_filter_, sizeof(float) * lap_size);
    assert(cudaState == cudaSuccess);

    //cout << "lap_filter = " << lap_filter << endl;
    //cout << "d_lap_filter_ = " << d_lap_filter_ << endl;

    // copy laplacian filter from host to device memory
    //cudaState = cudaMemcpy(d_lap_filter_, lap_filter, sizeof(float) * lap_size, cudaMemcpyHostToDevice);
    cudaState = cudaMemcpy(d_lap_filter_, lap_filter, sizeof(float) * lap_size, cudaMemcpyHostToDevice);
    //cout << cudaGetErrorString(cudaState) << endl;
    assert(cudaState == cudaSuccess);
    cout << "Laplacian filter created" << endl;

    float sig = 2 * gau_sig * gau_sig;
    float sum = 0.f;

    // make gau_rad = 5, gauss filter
    // gau_size = 11 * 11;
    for(int i = -gau_rad; i <= gau_rad; ++i)             // row
        for(int j = -gau_rad; j <= gau_rad; ++j)         // col
        {
            float val = i * i + j * j;
            val = exp(-val / sig) / (PI * sig);
            sum += val;
            int offset = ( i + gau_rad) * gau_wid + j;
            gau_filter[offset] = val;
        }

    for(int i = 0; i < gau_size; ++i)
        gau_filter[i] /= sum;

    // copy gauss filter from host to device
    cudaState = cudaMemcpy(d_gau_filter_, gau_filter,  sizeof(float) * gau_size, cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);

    cout << "Gauss filter created" << endl;

    //delete [] lap_filter;
    //delete [] gau_filter;

    cout << "Create LG filters Finished" << endl;
}

// input image is stored on global memory
// filterW is the width of laplacian filter
void WMap::laplafilter(float *d_out, float *d_in, int wid, int hei)
{
    int lap_wid = 2 * lap_rad + 1;
    // do laplacian filter laplacian kernel function because that the value should be adopted absolute value
    boxfilter(d_out, d_in, wid, hei, d_lap_filter_, lap_wid);
}

/*
// input image is stored on 2D Pitch
void WMap::laplafilter(float *d_out, float *d_in, size_t pitch, int wid, int hei)
{
    // laplacian filter width
    int lap_wid = 2 * lap_rad + 1;
    //boxfilter(d_out, d_in, pitch, wid, hei, d_lap_filter_, lap_wid);
}
*/

// input image is stored on global memory
// get the saliency map after doing gauss filter
void WMap::gaussfilter(float *d_out, float *d_in, int wid, int hei)
{
    const int gau_wid = 2 * gau_rad + 1;
    boxfilter(d_out, d_in, wid, hei, d_gau_filter_, gau_wid);
}

// compare the two map to generate the weight map
void WMap::weightmap(float *d_outA, float *d_outB, float *d_inA, float *d_inB, int wid, int hei)
{
    cudaError_t cudaState = cudaSuccess;
    // do this compare on the compare kernel function

    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + threadPerBlock.x - 1) / threadPerBlock.x;
    blockPerGrid.y = (hei + threadPerBlock.y - 1) / threadPerBlock.y;

    compare_kernel<<<blockPerGrid, threadPerBlock>>>(d_outA, d_outB, d_inA, d_inB, wid, hei);

    cudaState = cudaGetLastError();
    if(cudaState != cudaSuccess)
        cout << "In function Weight map generatioin : " << cudaGetErrorString(cudaState) << endl;
}

// get the final weight map based on weighted map and guided filter
void WMap::resultmap(float *d_out, float *d_inI, float *d_inP, int wid, int hei, int rad, float eps)
{
    guidedfilter(d_out, d_inI, d_inP, wid, hei);
}

