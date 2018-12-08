/**
 * @author smh
 * @date 2018.12.06
 *   三思而后言
 */

#include "RDLWavelet.h"

const int BLOCKSIZE = 16;

__constant__ float SincKernel[24];

#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__)
#define INDX(r, c, w) ((r) * (w) + (c))

inline void __cudaCheckError(cudaError_t err, const char* filename, const int line)
{
    if(err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << ": in  " << filename << ". At " << line << " line." << std::endl;
    }
}

int iDiv(int a, int b)
{
    if(a % b == 0)
        return a / b;
    else
        return a / b + 1;
}

RDLWavelet::RDLWavelet(int r, int c, int d) : rows_(r), cols_(c), dir_(d), cudaStatus_(cudaSuccess)
{
    // Copy Sinc kernel into device constant memory
    float Sinc[24] = {
                       -0.0110, 0.0452, -0.1437, 0.8950, 0.2777, -0.0812, 0.0233, -0.0158,
                       -0.0105, 0.0465, -0.1525, 0.6165, 0.6165, -0.1525, 0.0465, -0.0105,
                       -0.0053, 0.0233, -0.0812, 0.2777, 0.8950, -0.1437, 0.0452, -0.0110
                     };
    cudaCheckError(cudaMemcpyToSymbol(SincKernel, Sinc, sizeof(float) * 24));

    // pre-allocate all needed memory on host
    tempMatA_ = cv::Mat::zeros(cv::Size(c, r), CV_32FC1);
    tempMatB_ = cv::Mat::zeros(cv::Size(c, r), CV_32FC1);
    assert(!tempMatA_.empty());
    assert(!tempMatB_.empty());
    assert(tempMatA_.isContinuous());
    assert(tempMatB_.isContinuous());
    h_tempOutA_ = (float *)tempMatA_.data;
    h_tempOutB_ = (float *)tempMatB_.data;

    // pre-allocate all needed memory on device
    cudaCheckError(cudaMalloc(&d_imgIn_, sizeof(float) * r * c));
    cudaCheckError(cudaMalloc(&d_imgOut_, sizeof(float) * r * c));
    cudaCheckError(cudaMalloc(&d_Sinc_, sizeof(float) * r * c * 4));  // 插值后，变大4倍
    cudaCheckError(cudaMalloc(&d_tempA_, sizeof(float) * r * c));
    cudaCheckError(cudaMalloc(&d_tempB_, sizeof(float) * r * c));
    cudaCheckError(cudaMalloc(&d_tempC_, sizeof(float) * r * c));
    cudaCheckError(cudaMalloc(&d_tempD_, sizeof(float) * r * c));

    // create CUDA event
    cudaCheckError(cudaEventCreate(&startEvent_));
    cudaCheckError(cudaEventCreate(&stopEvent_));

    // CUDA thread hirearchy
    threadPerBlock_.x = BLOCKSIZE;
    threadPerBlock_.y = BLOCKSIZE;
    blockPerGrid_.x = iDiv(c, BLOCKSIZE);
    blockPerGrid_.y = iDiv(r, BLOCKSIZE);

    // CUDA stream
    cudaCheckError(cudaStreamCreate(&CudaStream_));    // used for hide data transfer latency with kernel computing
}

RDLWavelet::~RDLWavelet()
{
    if(d_imgIn_ != nullptr)
    {
        cudaFree(d_imgIn_);
        d_imgIn_ = nullptr;
    }
    if(d_imgOut_ != nullptr)
    {
        cudaFree(d_imgOut_);
        d_imgOut_ = nullptr;
    }
    if(d_Sinc_ != nullptr)
    {
        cudaFree(d_Sinc_);
        d_Sinc_ = nullptr;
    }
    if(d_tempA_ != nullptr)
    {
        cudaFree(d_tempA_);
        d_tempA_ = nullptr;
    }
    if(d_tempB_ != nullptr)
    {
        cudaFree(d_tempB_);
        d_tempB_ = nullptr;
    }
    if(d_tempC_ != nullptr)
    {
        cudaFree(d_tempC_);
        d_tempC_ = nullptr;
    }
    if(d_tempD_ != nullptr)
    {
        cudaFree(d_tempD_);
        d_tempD_ = nullptr;
    }

    cudaEventDestroy(startEvent_);
    cudaEventDestroy(stopEvent_);
}

void RDLWavelet::setParams(int d)
{
    while (d != 4)
    {
        std::cout << "Only 4 direction is supported now. Input again: " << std::endl;
        std::cin >> d;
    }
    dir_ = d;
}

__global__ void CudaSincInterpolation(float *d_imgOut, const float * __restrict__ d_imgIn, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   // idx-th col
    int idy = threadIdx.y + blockIdx.y * blockDim.y;   // idy-th row

    if(idx >= cols || idy >= rows)
        return ;

    d_imgOut[INDX(idy, idx * 4, cols * 4)] = d_imgIn[INDX(idy, idx, cols)];

    float sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    for(int i = -3; i <= 4; ++i)
    {
        int x = idx + i;
        if(x < 0)
            x = -x + 2;
        if(x >= cols)
            x = (idx << 1) - x;   // 等价于 idx - (x - idx) = 2 * idx - x

        sum1 += d_imgIn[INDX(idy, x, cols)] * SincKernel[INDX(0, i + 4 - 1, 8)];
        sum2 += d_imgIn[INDX(idy, x, cols)] * SincKernel[INDX(1, i + 4 - 1, 8)];
        sum3 += d_imgIn[INDX(idy, x, cols)] * SincKernel[INDX(2, i + 4 - 1, 8)];
    }

    d_imgOut[INDX(idy, idx * 4 + 1, cols * 4)] = sum1;
    d_imgOut[INDX(idy, idx * 4 + 2, cols * 4)] = sum2;
    d_imgOut[INDX(idy, idx * 4 + 3, cols * 4)] = sum3;
}

__global__ void CudaSincInterpolationVertical(float *d_imgOut, const float * __restrict__ d_imgIn, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   // idx-th col
    int idy = threadIdx.y + blockIdx.y * blockDim.y;   // idy-th row

    if(idx >= cols || idy >= rows)
        return ;

    d_imgOut[INDX(idy * 4, idx, cols)] = d_imgIn[INDX(idy, idx, cols)];

    float sumV;
    for(int i = -3; i <= 4; ++i)
    {
        int y = idy + i;
        if(y < 0)
            y = -y + 2;
        if(y >= rows)
            y = (idy << 1) - y;   // 等价于 idx - (x - idx) = 2 * idx - x

        sumV += d_imgIn[INDX(y, idx, cols)] * SincKernel[INDX(0, i + 4 - 1, 8)];
    }
    d_imgOut[INDX(idy * 4 + 1, idx, cols)] = sumV;
    sumV = 0.0;

    for(int i = -3; i <= 4; ++i)
    {
        int y = idy + i;
        if(y < 0)
            y = -y + 2;
        if(y >= rows)
            y = (idy << 1) - y;

        sumV += d_imgIn[INDX(y, idx, cols)] * SincKernel[INDX(1, i + 4 - 1, 8)];
    }
    d_imgOut[INDX(idy * 4 + 2, idx, cols)] = sumV;
    sumV = 0;

    for(int i = -3; i <= 4; ++i)
    {
        int y = idy + i;
        if(y < 0)
            y = -y + 2;
        if(y >= rows)
            y = (idy << 1) - y;   // 等价于 idx - (x - idx) = 2 * idx - x

        sumV += d_imgIn[INDX(y, idx, cols)] * SincKernel[INDX(2, i + 4 - 1, 8)];
    }
    d_imgOut[INDX(idy * 4 + 3, idx, cols)] = sumV;
}

// Three below kernels are equavilent to: special image filtering
// Horizontal Predict: copyMakeBorder + 2D sum filtering
// rows should be greater than 2
// cols is the number of cols before Sinc Interpolation! ! !
// d_imgIn: is the data after Sinc Interpolation ! ! !
// d_imgOut: is same size as the original input image
__global__ void CudaHorizontalPredict(float *d_imgOut, const float * __restrict__ d_imgIn, const float * __restrict__ d_SincIn, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   // idx-th col
    int idy = threadIdx.y + blockIdx.y * blockDim.y;   // idy-th row

    if(idx >= cols || idy >= rows)
        return ;

    float scale = 1.0 / ((4 * 2 + 1) * 2);
    float sumF = 0.0;
    //int x = 4 * idx + 4, y = idy;   // no need to + 4 to x, because no border is padded on the left
    int x = 4 * idx, y = idy;
    // 只考虑rowPtrUp部分
    for (int k = -4; k <= 4; ++k)
    {
        x += k;
        if(x < 0)     // not need actually, because x >= 4 >> x +k >= 0
            x = -x;   // reflect 101 mode
        if(x >= 4 * cols)
            x = 2 * 4 * idx - x;   // same as reasons in aboce Sinc Interpolation
        if(y == 0)            // same as reflect_101
            y += 2;

        sumF += d_SincIn[INDX(y - 1, x, cols * 4)];
    }
    // 只考虑rowPtrDown部分
    for(int k = -4; k <= 4; ++k)
    {
        x += k;
        if(x < 0)     // not need actually, because x >= 4 >> x +k >= 0
            x = -x;   // reflect 101 mode
        if(x >= 4 * cols)
            x = 2 * 4 * idx - x;   // same as reasons in aboce Sinc Interpolation
        if(y == rows - 1)            // same as reflect_101
            y -= 2;

        sumF += d_SincIn[INDX(y + 1, x, cols * 4)];
    }

    d_imgOut[INDX(idy, idx, cols)] = d_imgIn[INDX(idy, idx, cols)] - sumF * scale;
}
__global__ void CudaVerticalPredict(float *d_imgOut, const float * __restrict__ d_imgIn, const float * __restrict__ d_SincIn, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   // idx-th col
    int idy = threadIdx.y + blockIdx.y * blockDim.y;   // idy-th row

    if(idx >= cols || idy >= rows)
        return ;

    float scale = 1.0 / ((4 * 2 + 1) * 2);
    float sumF = 0.0;
    //int x = 4 * idx + 4, y = idy;   // no need to + 4 to x, because no border is padded on the left
    int x = idx, y = 4 * idy;
    // 只考虑rowPtrUp部分
    for (int k = -4; k <= 4; ++k)
    {
        y += k;
        if(y < 0)     // not need actually, because x >= 4 >> x +k >= 0
            y = -y;   // reflect 101 mode
        if(y >= 4 * rows)
            y = 2 * 4 * idy - y;   // same as reasons in aboce Sinc Interpolation
        if(x == 0)            // same as reflect_101
            x += 2;

        sumF += d_SincIn[INDX(y, x - 1, cols)];
    }
    // 只考虑rowPtrDown部分
    for(int k = -4; k <= 4; ++k)
    {
        y += k;
        if(y < 0)     // not need actually, because x >= 4 >> x +k >= 0
            y = -y;   // reflect 101 mode
        if(y >= 4 * rows)
            y = 2 * 4 * idy - y;   // same as reasons in aboce Sinc Interpolation
        if(x == cols - 1)            // same as reflect_101
            x -= 2;

        sumF += d_SincIn[INDX(y, x + 1, cols)];
    }

    d_imgOut[INDX(idy, idx, cols)] = d_imgIn[INDX(idy, idx, cols)] - sumF * scale;
}

__global__ void CudaHorizontalUpdate(float *d_imgOut, const float * __restrict__ d_imgIn, const float * __restrict__ d_SincIn, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   // idx-th col
    int idy = threadIdx.y + blockIdx.y * blockDim.y;   // idy-th row

    if(idx >= cols || idy >= rows)
        return ;

    float scale = 1.0 / ((4 * 2 + 1) * 2);
    float sumF = 0.0;
    //int x = 4 * idx + 4, y = idy;   // no need to + 4 to x, because no border is padded on the left
    int x = 4 * idx, y = idy;
    // 只考虑rowPtrUp部分
    for (int k = -4; k <= 4; ++k)
    {
        x += k;
        if(x < 0)     // not need actually, because x >= 4 >> x +k >= 0
            x = -x;   // reflect 101 mode
        if(x >= 4 * cols)
            x = 2 * 4 * idx - x;   // same as reasons in aboce Sinc Interpolation
        if(y == 0)            // same as reflect_101
            y += 2;

        sumF += d_SincIn[INDX(y - 1, x, cols * 4)];
    }
    // 只考虑rowPtrDown部分
    for(int k = -4; k <= 4; ++k)
    {
        x += k;
        if(x < 0)     // not need actually, because x >= 4 >> x +k >= 0
            x = -x;   // reflect 101 mode
        if(x >= 4 * cols)
            x = 2 * 4 * idx - x;   // same as reasons in aboce Sinc Interpolation
        if(y == rows - 1)            // same as reflect_101
            y -= 2;

        sumF += d_SincIn[INDX(y + 1, x, cols * 4)];
    }

    d_imgOut[INDX(idy, idx, cols)] = d_imgIn[INDX(idy, idx, cols)] + sumF * scale;
}
__global__ void CudaVerticalUpdate(float *d_imgOut, const float * __restrict__ d_imgIn, const float * __restrict__ d_SincIn, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   // idx-th col
    int idy = threadIdx.y + blockIdx.y * blockDim.y;   // idy-th row

    if(idx >= cols || idy >= rows)
        return ;

    float scale = 1.0 / ((4 * 2 + 1) * 2);
    float sumF = 0.0;
    //int x = 4 * idx + 4, y = idy;   // no need to + 4 to x, because no border is padded on the left
    int x = idx, y = 4 * idy;
    // 只考虑rowPtrUp部分
    for (int k = -4; k <= 4; ++k)
    {
        y += k;
        if(y < 0)     // not need actually, because x >= 4 >> x +k >= 0
            y = -y;   // reflect 101 mode
        if(y >= 4 * cols)
            y = 2 * 4 * idy - y;   // same as reasons in aboce Sinc Interpolation
        if(x == 0)            // same as reflect_101
            x += 2;

        sumF += d_SincIn[INDX(y, x - 1, cols)];
    }
    // 只考虑rowPtrDown部分
    for(int k = -4; k <= 4; ++k)
    {
        y += k;
        if(y < 0)     // not need actually, because x >= 4 >> x +k >= 0
            y = -y;   // reflect 101 mode
        if(y >= 4 * rows)
            y = 2 * 4 * idy - y;   // same as reasons in aboce Sinc Interpolation
        if(x == cols - 1)            // same as reflect_101
            x -= 2;

        sumF += d_SincIn[INDX(y, x + 1, cols)];
    }

    d_imgOut[INDX(idy, idx, cols)] = d_imgIn[INDX(idy, idx, cols)] + sumF * scale;
}

__global__ void CudaInverseHorizontalUpdate(float *d_imgOut, const float * __restrict__ d_imgBase, const float * __restrict__ d_SincDetail, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   // idx-th col
    int idy = threadIdx.y + blockIdx.y * blockDim.y;   // idy-th row

    if(idx >= cols || idy >= rows)
        return ;

    float scale = 1.0 / ((4 * 2 + 1) * 2);
    float sumF = 0.0;
    //int x = 4 * idx + 4, y = idy;   // no need to + 4 to x, because no border is padded on the left
    int x = 4 * idx, y = idy;
    // 只考虑rowPtrUp部分
    for (int k = -4; k <= 4; ++k)
    {
        x += k;
        if(x < 0)     // not need actually, because x >= 4 >> x +k >= 0
            x = -x;   // reflect 101 mode
        if(x >= 4 * cols)
            x = 2 * 4 * idx - x;   // same as reasons in aboce Sinc Interpolation
        if(y == 0)            // same as reflect_101
            y += 2;

        sumF += d_SincDetail[INDX(y - 1, x, cols * 4)];
    }
    // 只考虑rowPtrDown部分
    for(int k = -4; k <= 4; ++k)
    {
        x += k;
        if(x < 0)     // not need actually, because x >= 4 >> x +k >= 0
            x = -x;   // reflect 101 mode
        if(x >= 4 * cols)
            x = 2 * 4 * idx - x;   // same as reasons in aboce Sinc Interpolation
        if(y == rows - 1)            // same as reflect_101
            y -= 2;

        sumF += d_SincDetail[INDX(y + 1, x, cols * 4)];
    }

    d_imgOut[INDX(idy, idx, cols)] = d_imgBase[INDX(idy, idx, cols)] - sumF * scale;
}
__global__ void CudaInverseVerticalUpdate(float *d_imgOut, const float * __restrict__ d_imgBase, const float * __restrict__ d_SincDetail, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   // idx-th col
    int idy = threadIdx.y + blockIdx.y * blockDim.y;   // idy-th row

    if(idx >= cols || idy >= rows)
        return ;

    float scale = 1.0 / ((4 * 2 + 1) * 2);
    float sumF = 0.0;
    //int x = 4 * idx + 4, y = idy;   // no need to + 4 to x, because no border is padded on the left
    int x = idx, y = 4 * idy;
    // 只考虑rowPtrUp部分
    for (int k = -4; k <= 4; ++k)
    {
        y += k;
        if(y < 0)     // not need actually, because x >= 4 >> x +k >= 0
            y = -y;   // reflect 101 mode
        if(y >= 4 * rows)
            y = 2 * 4 * idy - y;   // same as reasons in aboce Sinc Interpolation
        if(x == 0)            // same as reflect_101
            x += 2;

        sumF += d_SincDetail[INDX(y, x - 1, cols)];
    }
    // 只考虑rowPtrDown部分
    for(int k = -4; k <= 4; ++k)
    {
        y += k;
        if(y < 0)     // not need actually, because x >= 4 >> x +k >= 0
            y = -y;   // reflect 101 mode
        if(y >= 4 * rows)
            y = 2 * 4 * idy - y;   // same as reasons in aboce Sinc Interpolation
        if(x == cols - 1)            // same as reflect_101
            x -= 2;

        sumF += d_SincDetail[INDX(y, x + 1, cols)];
    }

    d_imgOut[INDX(idy, idx, cols)] = d_imgBase[INDX(idy, idx, cols)] - sumF * scale;
}

// Three below functions are based on GPU data pointer, so no parameters
void RDLWavelet::HorizontalPredict(float *d_imgOut, const float * __restrict__ d_imgIn)
{
    CudaSincInterpolation<<<blockPerGrid_, threadPerBlock_>>>(d_Sinc_, d_imgIn, rows_, cols_);
    //CudaHorizontalPredict<<<blockPerGrid_, threadPerBlock_>>>(d_imgOut_, d_imgIn_, d_Sinc_, rows_, cols_);
    CudaHorizontalPredict<<<blockPerGrid_, threadPerBlock_>>>(d_imgOut, d_imgIn, d_Sinc_, rows_, cols_);

    cudaCheckError(cudaPeekAtLastError());
}
void RDLWavelet::VerticalPredict(float *d_imgOut, const float *d_imgIn)
{
    CudaSincInterpolationVertical<<<blockPerGrid_, threadPerBlock_>>>(d_Sinc_, d_imgIn, rows_, cols_);
    CudaVerticalPredict<<<blockPerGrid_, threadPerBlock_>>>(d_imgOut, d_imgIn, d_Sinc_, rows_, cols_);

    cudaCheckError(cudaPeekAtLastError());
}

void RDLWavelet::HorizontalUpdate(float *d_imgOut, const float * __restrict__ d_imgDetail, const float * __restrict__ d_imgIn)
{
    CudaSincInterpolation<<<blockPerGrid_, threadPerBlock_>>>(d_Sinc_, d_imgDetail, rows_, cols_);
    CudaHorizontalUpdate<<<blockPerGrid_, threadPerBlock_>>>(d_imgOut, d_imgIn, d_Sinc_, rows_, cols_);

    cudaCheckError(cudaPeekAtLastError());
}
void RDLWavelet::VerticalUpdate(float *d_imgOut, const float *d_imgDetail, const float *d_imgIn)
{
    CudaSincInterpolationVertical<<<blockPerGrid_, threadPerBlock_>>>(d_Sinc_, d_imgDetail, rows_, cols_);
    CudaVerticalUpdate<<<blockPerGrid_, threadPerBlock_>>>(d_imgOut, d_imgIn, d_Sinc_, rows_, cols_);

    cudaCheckError(cudaPeekAtLastError());
}

void RDLWavelet::InverseHorizontalUpdate(float *d_imgOut, const float * __restrict__ d_imgBase, const float * __restrict__ d_imgDetail)
{
    // TODO
}
void RDLWavelet::InverseVerticalUpdate(float *d_imgOut, const float * __restrict__ d_imgBase, const float * __restrict__ d_imgDetail)
{
    CudaSincInterpolationVertical<<<blockPerGrid_, threadPerBlock_>>>(d_Sinc_, d_imgDetail, rows_, cols_);
    CudaInverseVerticalUpdate<<<blockPerGrid_, threadPerBlock_>>>(d_imgOut, d_imgBase, d_Sinc_, rows_, cols_);

    cudaCheckError(cudaPeekAtLastError());
}

void RDLWavelet::doRDLWavelet(std::vector<cv::Mat> &imgOuts, const cv::Mat &imgIn)
{
    h_imgIn_ = (float *)imgIn.data;
    cudaCheckError(cudaMemcpy(d_imgIn_, h_imgIn_, sizeof(float) * rows_ * cols_, cudaMemcpyHostToDevice));

    // Step 1: Horizontal Predict & Update
    // TODO


    // Step 2: Vertical Predict & Update
}

//void RDLWavelet::doInverseRDLWavelet(cv::Mat &imgOut, std::vector<cv::Mat> &imgIns)
void RDLWavelet::doInverseRDLWavelet(cv::Mat &imgOut)
{
    assert(!imgOut.empty());
    assert(imgOut.isContinuous());

    float *h_imgOut = (float *)imgOut.data;

    // Step 1: Inverse Vertical Transform, output, base layer, detail layer
    // TODO

    // Step 2: Inverse Horizontal Transform, output, base layer, detail layer

    // Step 3: copy final data to host
}
