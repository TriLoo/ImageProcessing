/**
 * @author smh
 * @date 2018.12.10
 *
 * @brief CUDA Implementation of Guided filter for grayscale image.
 *   没有人能原谅你，除了你自己。
 */

#include "GuidedFilter.h"

//texture<float, cudaTextureType2D> imgTexI, imgTexP;
//texture<float, cudaTextureType1D> texInI, texInP;
texture<float, cudaTextureType2D> texInI, texInP;

namespace IVFusion
{
    GuidedFilter::GuidedFilter(int r, int c): rows_(r), cols_(c)
    {
        cudaCheckError(cudaMalloc(&d_tempA_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempB_, sizeof(float) * r * c));

        blockPerGrid = dim3(iDiv(c, BLOCKSIZE), iDiv(r, BLOCKSIZE));
        threadPerBlock = dim3(BLOCKSIZE, BLOCKSIZE);
    }

    GuidedFilter::~GuidedFilter()
    {
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
    }

    void GuidedFilter::bindTexture(float *d_imgInI, float *d_imgInP)
    {
        cudaChannelFormatDesc cudaChannel = cudaCreateChannelDesc<float>();

        //cudaCheckError(cudaBindTexture(NULL, texInI, d_imgInI, cudaChannel, sizeof(float) * rows_ * cols_));
        //cudaCheckError(cudaBindTexture(NULL, texInP, d_imgInP, cudaChannel, sizeof(float) * rows_ * cols_));
        cudaCheckError(cudaBindTexture2D(NULL, texInI, d_imgInI, cudaChannel, sizeof(float) * cols_, rows_, sizeof(float) * cols_));
        cudaCheckError(cudaBindTexture2D(NULL, texInP, d_imgInP, cudaChannel, sizeof(float) * cols_, rows_, sizeof(float) * cols_));
    }

    void GuidedFilter::releaseTexture()
    {
        cudaCheckError(cudaUnbindTexture(texInI));
        cudaCheckError(cudaUnbindTexture(texInP));
    }

    // row cumulate sum, based on global memory
    __global__ void d_boxfiler_x(float *d_out_, const float * d_in_, int rows, int cols, int rad)
    {
        int y = blockIdx.x * blockDim.x + threadIdx.x;

        if(y > rows)
            return ;

        float scale = 1.0f / (float)((rad << 1) + 1.0f);

        const float *d_in = &d_in_[y * cols];
        float *d_out = &d_out_[y * cols];

        // do left edge
        float t = d_in[0] * rad;
        for(int x = 0; x < (rad + 1); ++x)
            t += d_in[x];
        d_out[0] = t * scale;

        // do main loop
        for(int x = (rad + 1); x < (cols - rad); ++x)
        {
            t += d_in[x+rad];
            t -= d_in[x - rad - 1];
            d_out[x] = t * scale;
        }

        // do right edge
        for(int x= cols - rad; x < cols; ++x)
        {
            t += d_in[cols - 1];
            t -= d_in[x - rad - 1];
            d_out[x] = x * scale;
        }
    }
    // row cumulate sum, based on 2D texture memory
    // boxfilter on image I
    __global__ void
    d_box_I_x(float* d_out, int rows, int cols, int rad)
    {
        unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

        float scale = 1.0f / (float)((rad << 1) + 1.0f);

        if (y < rows)
        {
            float t = 0.0f;
            for (int x = -rad; x <= rad; ++x)
            {
                t += tex2D(texInI, x, y);
            }

            d_out[y * cols] = t * scale;

            for (int x = 1; x < cols; ++x)
            {
                t += tex2D(texInI, x + rad, y);
                t -= tex2D(texInI, x - rad - 1, y);
                d_out[y * cols + x] = t * scale;
            }
        }
    }
    // boxfilter on image P
    __global__ void
    d_box_P_x(float* d_out, int rows, int cols, int rad)
    {
        unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
        float scale = 1.0f / (float)((rad << 1) + 1.0f);

        if (y < rows)
        {
            float t = 0.0f;
            for (int x = -rad; x <= rad; ++x)
            {
                t += tex2D(texInP, x, y);
            }

            d_out[y * cols] = t * scale;

            for (int x = 1; x < cols; ++x)
            {
                t += tex2D(texInP, x + rad, y);
                t -= tex2D(texInP, x - rad - 1, y);
                d_out[y * cols + x] = t * scale;
            }
        }
    }

    __global__ void d_boxfilter_y(float *d_out_, const float * __restrict__ d_in_, const int rows, const int cols, const int rad)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if(x >= cols)
            return ;

        const float *d_in = &d_in_[x];
        float *d_out = &d_out_[x];
        float scale = 1.0f / (float)((rad << 1) + 1.0f);

        float t = d_in[0] * rad;
        for(int y = 0; y < (rad + 1); ++y)
            t += d_in[y * cols];

        d_out[0] = t * scale;

        // do up edge
        for(int y = 1; y < rad + 1; ++y)
        {
            t += d_in[(y + rad) * cols];
            t -= d_in[0];
            d_out[y * cols] = t * scale;
        }

        // do main loop
        for(int y = (1 + rad); y < (rows - rad); ++y)
        {
            t += d_in[(y + rad) * cols];
            t -= d_in[(y - rad - 1) * cols];
            d_out[y * cols] = t * scale;
        }

        // do down edge
        for(int y = rows - rad; y < rows; ++y)
        {
            t += d_in[(rows - 1) * cols];
            t -= d_in[((y - rad) * cols) - cols];
            d_out[y * cols] = t * scale;
        }
    }

    // boxfilter to calculate corrI: corrI = fmean( I .* I )
    __global__ void
    d_square_box_I_x(float* d_out, int rows, int cols, int rad)
    {
        unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
        float scale = 1.0f / (float)((rad << 1) + 1.0f);

        if (y < rows)
        {
            float t = 0.0f;
            for (int x = -rad; x <= rad; ++x)
            {
                t += tex2D(texInI, x, y) * tex2D(texInI, x, y);
            }

            d_out[y * cols] = t * scale;

            for (int x = 1; x < cols; ++x)
            {
                t += tex2D(texInI, x + rad, y) * tex2D(texInI, x + rad, y);
                t -= tex2D(texInI, x - rad - 1, y) * tex2D(texInI, x - rad - 1, y);
                d_out[y * cols + x] = t * scale;
            }
        }
    }

    // kernels to calculate A
    // - corrIp
    __global__ void d_box_corrIp_x(float *d_out, int rows, int cols, const int rad)
    {
        unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
        float scale = 1.0f / (float)((rad << 1) + 1.0f);

        if (y < rows)
        {
            float t = 0.0f;
            for (int x = -rad; x <= rad; ++x)
                t += tex2D(texInP, x, y) * tex2D(texInI, x, y);
            d_out[y * cols] = t * scale;

            for (int x = 1; x < cols; ++x)
            {
                t += tex2D(texInP, x + rad, y) *  tex2D(texInI, x + rad, y);
                t -= tex2D(texInP, x - rad - 1, y) * tex2D(texInI, x - rad - 1, y);

                d_out[y * cols + x] = t * scale;
            }
        }
    }

    __global__ void d_box_corrIp_y(float *d_out, float *d_imgInI_, float *d_imgInP_, int rows, int cols, const int rad)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x < cols)
        {
            float *d_imgInI = &d_imgInI_[x];
            float *d_imgInP = &d_imgInP_[x];

            float t = d_imgInI[0] * d_imgInP[0] * rad;
            float scale = 1.0f / (float)((rad << 1) + 1.0f);

            for (int y = 0; y < (rad + 1); y++)
                t += d_imgInI[y * cols] * d_imgInP[y * cols];

            d_out[0] = t * scale;

            // do up edge
            for (int y = 1; y < rad + 1; y++)
            {
                t += d_imgInI[y * cols] * d_imgInP[y * cols];
                t -= d_imgInI[0] * d_imgInP[0];
                d_out[y * cols] = t * scale;
            }

            // do main loop
            for (int y = (1 + rad); y < (rows - rad); y++)
            {
                t += d_imgInI[(y + rad) * cols] * d_imgInP[(y + rad) * cols];
                t -= d_imgInI[(y - rad - 1) * cols] * d_imgInP[(y - rad - 1) * cols];
                d_out[y * cols] = t * scale;
            }

            // do right edge
            for (int y = rows - rad; y < rows; y++)
            {
                t += d_imgInI[(rows - 1) * cols] * d_imgInP[(rows - 1) * cols];
                t -= d_imgInI[(y - rad - 1) * cols] * d_imgInP[(y - rad - 1) * cols];
                d_out[y * cols] = t * scale;
            }
        }
    }

    // - calculate A
    // aB, aG, aR are equal to corrIp
    __global__ void calculateA(float *d_out, float *d_corrIp, float *d_corrI, float *d_meanI, float *d_meanP, int rows, int cols, double eps)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;

        if (idx >= cols || idy >= rows)
            return ;

        int index = idx + idy * cols;

        float val_P = d_meanP[index];              // val_P: the value from meanP
        float val_I = d_meanI[index];

        //float var_I = corrI[index] - val_I * val_I + eps;
        float var_I = d_corrI[index] - val_I * val_P + eps;
        float covIp = d_corrIp[index] - val_I * val_P;
        d_out[index] = covIp / var_I;
    }

    // kernels to calculate B
    __global__ void calculateB(float *d_out, float *d_inA, float *d_meanI, float *d_meanP, int rows, int cols)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;

        if (idx >= cols || idy >= rows)
            return ;

        int index = idx + idy * cols;

        d_out[index] = d_meanP[index] - d_inA[index] * d_meanI[index];
    }

    // kernels to calculate Q (Results)
    // inB: the mean of B
    // imgIO: the input (meanB) and the Output(Q)
    __global__ void calculateQ(float *d_out, float *d_meanA, int rows, int cols)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;

        if (idx >= cols || idy >= rows)
            return ;

        int index = idx + idy * cols;

        d_out[index] += tex2D(texInI, idx, idy) * d_meanA[index];
    }

    void GuidedFilter::doGuidedFilter(float *d_detail, float *d_base, const cv::Mat &imgInI, const cv::Mat &imgInP,
                                      int rad1, double eps1, int rad2, double eps2)
    {

    }

    void GuidedFilter::doGuidedFilter(float *d_imgOut, const float *d_imgInI, float *d_imgInP, cudaStream_t cs, int rad, double eps)
    {

    }
}     // namespace IVFusion

