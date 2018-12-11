/**
* @author smh
 * @date 2018.12.10
 *
 * @brief Implementation of redundant directional lifting-based wavelet
 *   临渊羡鱼，不若退而结网。
 */

#include "RDLWavelet.h"

__constant__ float SincKernel[24];

namespace IVFusion
{
    RDLWavelet::RDLWavelet(int r, int c, int d, cudaStream_t cs): rows_(r), cols_(c), dir_(d)
    {
        float Sinc[24] = {
                       -0.0110, 0.0452, -0.1437, 0.8950, 0.2777, -0.0812, 0.0233, -0.0158,
                       -0.0105, 0.0465, -0.1525, 0.6165, 0.6165, -0.1525, 0.0465, -0.0105,
                       -0.0053, 0.0233, -0.0812, 0.2777, 0.8950, -0.1437, 0.0452, -0.0110
                     };
        cudaCheckError(cudaMemcpyToSymbol(SincKernel, Sinc, sizeof(float) * 8 * 3));

        // set the thread hirearchy
        threadPerBlock_ = dim3(BLOCKSIZE, BLOCKSIZE);
        blockPerGrid_.x = iDiv(c, BLOCKSIZE);
        blockPerGrid_.y = iDiv(r, BLOCKSIZE);
        // OR
        //blockPerGrid_ = dim3(iDiv(c, BLOCKSIZE), iDiv(r, BLOCKSIZE));

        // temporary allocated memory
        cudaCheckError(cudaMalloc(&d_Sinc_, sizeof(float) * r * c * 4));
        cudaCheckError(cudaMalloc(&d_temp_, sizeof(float) * r * c));

        // pre-defined cuda stream
        cudaCheckError(cudaStreamCreate(&stream1_));
        stream1_ = cs;
    }

    RDLWavelet::~RDLWavelet()
    {
        if(d_Sinc_ != nullptr)
        {
            cudaFree(d_Sinc_);
            d_Sinc_ = nullptr;
        }

        if(d_temp_ != nullptr)
        {
            cudaFree(d_temp_);
            d_temp_ = nullptr;
        }
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
        CudaSincInterpolation<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_Sinc_, d_imgIn, rows_, cols_);
        //CudaHorizontalPredict<<<blockPerGrid_, threadPerBlock_>>>(d_imgOut_, d_imgIn_, d_Sinc_, rows_, cols_);
        CudaHorizontalPredict<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_imgOut, d_imgIn, d_Sinc_, rows_, cols_);

        cudaCheckError(cudaPeekAtLastError());
    }
    void RDLWavelet::VerticalPredict(float *d_imgOut, const float *d_imgIn)
    {
        CudaSincInterpolationVertical<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_Sinc_, d_imgIn, rows_, cols_);
        CudaVerticalPredict<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_imgOut, d_imgIn, d_Sinc_, rows_, cols_);

        cudaCheckError(cudaPeekAtLastError());
    }

    void RDLWavelet::HorizontalUpdate(float *d_imgOut, const float * __restrict__ d_imgDetail, const float * __restrict__ d_imgIn)
    {
        CudaSincInterpolation<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_Sinc_, d_imgDetail, rows_, cols_);
        CudaHorizontalUpdate<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_imgOut, d_imgIn, d_Sinc_, rows_, cols_);

        cudaCheckError(cudaPeekAtLastError());
    }
    void RDLWavelet::VerticalUpdate(float *d_imgOut, const float *d_imgDetail, const float *d_imgIn)
    {
        CudaSincInterpolationVertical<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_Sinc_, d_imgDetail, rows_, cols_);
        CudaVerticalUpdate<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_imgOut, d_imgIn, d_Sinc_, rows_, cols_);

        cudaCheckError(cudaPeekAtLastError());
    }

    void RDLWavelet::InverseHorizontalUpdate(float *d_imgOut, const float * __restrict__ d_imgBase, const float * __restrict__ d_imgDetail)
    {
        CudaSincInterpolation<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_Sinc_, d_imgDetail, rows_, cols_);
        CudaInverseHorizontalUpdate<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_imgOut, d_imgBase, d_Sinc_, rows_, cols_);

        cudaCheckError(cudaPeekAtLastError());
    }
    void RDLWavelet::InverseVerticalUpdate(float *d_imgOut, const float * __restrict__ d_imgBase, const float * __restrict__ d_imgDetail)
    {
        CudaSincInterpolationVertical<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_Sinc_, d_imgDetail, rows_, cols_);
        CudaInverseVerticalUpdate<<<blockPerGrid_, threadPerBlock_, 0, stream1_>>>(d_imgOut, d_imgBase, d_Sinc_, rows_, cols_);

        cudaCheckError(cudaPeekAtLastError());
    }

    // all inputs: d_cD, d_cV, d_cH, d_cA, is allocated but not used.
    void RDLWavelet::doRDLWavelet(float *d_cD, float *d_cV, float *d_cH, float *d_cA, float *d_imgIn)
    {
        // Step 1: Horizontal Predict & Update
        HorizontalPredict(d_temp_, d_imgIn);                       // d_temp_     ->     H
        HorizontalUpdate(d_cA, d_temp_, d_imgIn);                  // d_tempB_    ->     L

        // Step 2: Vertical Predict & Update
        VerticalPredict(d_cD, d_temp_);                            // d_cD     ->      HH
        VerticalUpdate(d_cV, d_cD, d_temp_);                       // d_cV     ->      HL

        VerticalPredict(d_cH, d_cA);                               // d_cH       ->      LH
        VerticalUpdate(d_temp_, d_cH, d_cA);                       // d_temp_    ->      LL

        cudaCheckError(cudaMemcpy(d_cA, d_temp_, sizeof(float) * rows_ * cols_, cudaMemcpyDeviceToDevice));
        cudaCheckError(cudaPeekAtLastError());
    }

    //void RDLWavelet::doInverseRDLWavelet(cv::Mat &imgOut, std::vector<cv::Mat> &imgIns)
    void RDLWavelet::doInverseRDLWavelet(float *d_imgOut, float *d_cD, float *d_cV, float *d_cH, float *d_cA)
    {
        // Step 1: Inverse Vertical Transform, output, base layer, detail layer
        InverseVerticalUpdate(d_temp_, d_cV, d_cD);             // d_temp_    ->   Detail layer
        InverseVerticalUpdate(d_cV, d_cA, d_cH);                // d_tempC_     ->   Base layer

        // Step 2: Inverse Horizontal Transform, output, base layer, detail layer
        InverseHorizontalUpdate(d_imgOut, d_cV, d_temp_);         // d_imgOut     ->   Final fused image

        // Step 3: copy final data to host
        // cudaCheckError(cudaMemcpy(h_imgOut, d_cA, sizeof(float) * rows_ * cols_, cudaMemcpyDeviceToHost));

        cudaCheckError(cudaPeekAtLastError());
    }
}   // namespace IVFusion
