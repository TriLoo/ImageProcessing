//
// Author : smh - 2017.01.16
//

#include "GuidedFilter.h"
#include "helper_math.h"

#define BLOCKSIZE 16   // BLOCKSIZE * BLOCKSIZE threads per block

using namespace std;
using namespace cv;

texture<float4, cudaTextureType2D> rgbaTex;
cudaArray *rgbaIn_d, *rgbaOut_d;

void imgShow(Mat img)
{
    imshow("Temp", img);
    waitKey(0);
}

GFilter::GFilter(int r, int c) : row_(r), col_(c), rad_(45), eps_(0.000001)
{
    cudaEventCreate(&startEvent_);
    cudaEventCreateWithFlags(&stopEvent_, cudaEventBlockingSync);
}

GFilter::~GFilter()
{
    cudaEventDestroy(startEvent_);
    cudaEventDestroy(stopEvent_);
}

// Kernel functions
// __device__
// do boxfilter
__global__ void
d_boxfilter_rgb_x(float4* d_out, int row, int col, int rad)
{
    float scale = 1.0f / (float)((rad << 1) + 1.0f);
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < row)
    {
        float4 t = make_float4(0.0f);
        for (int x = -rad; x <= rad; ++x)
        {
            t += tex2D(rgbaTex, x, y);
        }

        d_out[y * col] = t * scale;

        for (int x = 1; x < col; ++x)
        {
            t += tex2D(rgbaTex, x + rad, y);
            t -= tex2D(rgbaTex, x - rad, y);
            d_out[y * col] = t * scale;
        }
    }
}

/*
__global__ void
d_boxfilter_rgb_y(float4 * d_out, int row, int col, int rad)
{
}
*/

__global__ void
d_boxfilter_rgb_y(float4* d_out, float4* d_in, const int row, const int col, const int rad)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if ( x >= col )
        return;

    d_in = &d_in[x];
    d_out = &d_out[x];

    float scale = 1.0f / (float)((rad << 1) + 1.0f);

    float4 t = make_float4(0.0f);

    t = d_in[0] * rad;

    for (int y = 0; y < (rad + 1); ++y)
    {
        t += d_in[y * col];
    }

    d_out[0] = t * scale;

    // do left edge
    for (int y = 1; y < rad + 1; ++y)
    {
        t += d_in[y * col];
        t -= d_in[0];
        d_out[y * col] = t * scale;
    }

    // do main loop
    for (int y = 1 + rad; y < (row - rad); ++y)
    {
        t += d_in[(y + rad) * col];
        t -= d_in[(y - rad) * col];
        d_out[y * col] = t * scale;
    }

    // do right edge
    for (int y = row - rad; y < row; ++y)
    {
        t += d_in[(row - 1) * col];
        t -= d_in[(y - row) * col - col];

        d_out [y * col] = t * scale;
    }
}

void GFilter::initTexture(float* data)
{
    float* tempH = new float [row_ * col_ * 4];
    float* tempD = data;
    float* tempSrc = tempH;
    const int size = row_ * col_;
    for (int i = 0; i < size; ++i)
    {
        *tempH++ = *tempD++;
        *tempH++ = *tempD++;
        *tempH++ = *tempD++;
        *tempH++ = 0.0;
    }

    // allocate the 2d Array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaCheckError(cudaMallocArray(&rgbaIn_d, &channelDesc, col_, row_));
    cudaCheckError(cudaMemcpyToArray(rgbaIn_d, 0, 0, tempSrc, size * 4, cudaMemcpyHostToDevice));

    // bind array to texture
    cudaCheckError(cudaBindTextureToArray(rgbaTex, rgbaIn_d, channelDesc));

    delete [] tempH;
}

void releaseTexture()
{
    //cudaUnbindTexture(rgbaTex);
    cudaCheckError(cudaFreeArray(rgbaIn_d));
    cudaCheckError(cudaFreeArray(rgbaOut_d));
}

void GFilter::restoreFromFloat4(float *out, float *in)
{
    float *tempIn = in;
    float *tempOut = out;

    for (int i = 0; i < row_; ++i)
        for (int j = 0; j < col_; ++j)
        {
            *tempOut++ = *tempIn++;
            *tempOut++ = *tempIn++;
            *tempOut++ = *tempIn++;
            ++tempIn;
        }

}

void GFilter::boxfilter(float *imgOut_d, const float *imgIn_d, int rad)
{
}

void GFilter::boxfilterTest(cv::Mat &imgOut, const cv::Mat &imgIn, int rad)
{
    float *dataInP = (float *)imgIn.data;
    initTexture(dataInP);
    float *dataOutP = (float *)imgOut.data;

    float4 *tempData, *outDataD;
    float *tempDataH = new float [row_ * col_ * sizeof(float4)];
    //cudaChannelFormatDesc channels = cudaCreateChannelDesc<float4>();
    cudaCheckError(cudaMalloc((void **)&tempData, sizeof(float4) * col_ * row_));
    cudaCheckError(cudaMalloc((void **)&outDataD, sizeof(float4) * col_ * row_));

    // use texture for horizontal pass
    d_boxfilter_rgb_x<<<row_ / BLOCKSIZE, BLOCKSIZE, 0>>>(tempData, row_, col_, rad);
    d_boxfilter_rgb_y<<<col_ / BLOCKSIZE, BLOCKSIZE, 0>>>(outDataD, tempData, row_, col_, rad);

    cudaCheckError(cudaMemcpy(tempDataH, outDataD, sizeof(float4) * row_ * col_, cudaMemcpyDeviceToHost));

    restoreFromFloat4(dataOutP, tempDataH);

    delete [] tempDataH;
}

void GFilter::boxfilterNpp(cv::Mat &imgOut, const cv::Mat &imgIn, int rad)
{
    assert(imgIn.isContinuous());
    const float* imgI_h = (const float*)imgIn.data;
    float* imgOut_h = (float *)imgOut.data;
    int pSrcStepBytes = col_ * sizeof(float) * imgIn.channels();

    int pStepBytes;
    Npp32f* imgIn_d = nppiMalloc_32f_C3(col_, row_, &pStepBytes);
    NppStatus stateNpp = NPP_SUCCESS;
    cudaError_t stateCUDA = cudaSuccess;
    NppiSize sizeROI;
    sizeROI.width = col_;
    sizeROI.height = row_;
    // Copy image from host to device
    stateCUDA = cudaMemcpy2D(imgIn_d, pStepBytes, imgI_h, pSrcStepBytes, pSrcStepBytes, row_, cudaMemcpyHostToDevice);
    assert(stateCUDA == cudaSuccess);
    Npp32f* imgOut_d = nppiMalloc_32f_C3(col_, row_, &pStepBytes);
    NppiSize oMaskSize = {16, 16};
    NppiPoint oAnchor = {oMaskSize.width/2, oMaskSize.height / 2};

    cudaEventRecord(startEvent_, 0);
    stateNpp = nppiFilterBoxBorder_32f_C3R(imgIn_d, pStepBytes, sizeROI, {0,0}, imgOut_d, pStepBytes, sizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    cudaEventRecord(stopEvent_, 0);
    //cudaEventSynchronize(stopEvent_);
    cudaEventElapsedTime(&elapsedTime_, startEvent_, stopEvent_);
    cout << "Only GPU Time: " << elapsedTime_ << "ms." << endl;
    if (stateNpp != NPP_SUCCESS)
    {
        nppiFree(imgIn_d);
        nppiFree(imgOut_d);
        exit(EXIT_FAILURE);
    }

    stateCUDA = cudaMemcpy2D(imgOut_h, pSrcStepBytes, imgOut_d, pStepBytes, pStepBytes, row_, cudaMemcpyDeviceToHost);
    assert(stateCUDA == cudaSuccess);
    cudaDeviceSynchronize();
    nppiFree(imgIn_d);
    nppiFree(imgOut_d);
}

void GFilter::gaussianfilter(float *imgOut_d, const float *imgIn_d, int rad, double sig)
{
}

// 输入图像是相同的  e.g. imgInI == imgInP
// color image guided filter
void GFilter::guidedfilterSingle(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
}

// 输入图像是不同的  e.g. imgInI != imgInP
void GFilter::guidedfilterDouble(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
}

void GFilter::guidedfilter(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    assert(imgInP.channels() == 3 && imgInI.channels() == 3);
    //const float *imgA = (float *)imgInI.data;
    //const float *imgB = (float *)imgInP.data;
    equal_to<const float*> T;
    if (T((float *)imgInI.data, (float*)imgInP.data))
        guidedfilterSingle(imgOut, imgInI, imgInI);
    else
        guidedfilterDouble(imgOut, imgInI, imgInP);
}

// Contrast Experiments: Guided Filter based on OpenCV
void GFilter::guidedfilterOpenCV(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    assert(imgInP.channels() == 3);
    if (rad_ == 0)
        setParams(16, 0.01);    // Image Enhancement

    Mat meanI, corrI, varI, meanP;
    boxFilter(imgInI, meanI, imgInI.depth(), Size(rad_, rad_));
    boxFilter(imgInI.mul(imgInI), corrI, imgInI.depth(), Size(rad_, rad_));
    boxFilter(imgInP, meanP, imgInP.depth(), Size(rad_, rad_));
    varI = corrI - meanI.mul(meanI);
    //imgShow(varI);

    vector<Mat> vecP(imgInP.channels()), vecI(imgInI.channels());
    vector<Mat> vecMeanI(imgInI.channels()), vecMeanP(imgInP.channels());
    split(imgInP, vecP);
    split(imgInI, vecI);
    split(meanP, vecMeanP);
    split(meanI, vecMeanI);

    Mat covIp, sameP, sameMeanP, meanA, meanB;
    vector<Mat> vecA(imgInI.channels());
#pragma unloop
    for (int i = 0; i < 3; ++i)
    {
        //vector<Mat> vecSameP{vecP[i], vecP[i], vecP[i]};
        //merge(vecSameP, sameP);
        //boxFilter(imgInI.mul(sameP), covIp, imgInI.depth(), Size(rad_, rad_));
        //vector<Mat> vecSameMeanP{vecMeanP[i], vecMeanP[i], vecMeanP[i]};
        //merge(vecSameMeanP, sameMeanP);
        cvtColor(vecP[i], sameP, CV_GRAY2BGR);         // use cvtColor to do the broadcast purpose, instead of above method
        cvtColor(vecMeanP[i], sameMeanP, CV_GRAY2BGR);
        boxFilter(imgInI.mul(sameP), covIp, imgInI.depth(), Size(rad_, rad_));
        covIp = covIp - meanI.mul(sameMeanP);

        Mat a = covIp / (varI + eps_);
        boxFilter(a, meanA, a.depth(), Size(rad_, rad_));
        //cout << "a.channels = " << a.channels() << endl;         // for test

        split(a, vecA);
        Mat b = vecMeanP[i] - (vecA[0].mul(vecMeanI[0]) + vecA[1].mul(vecMeanI[1]) + vecA[2].mul(vecMeanI[2]));
        boxFilter(b, meanB, b.depth(), Size(rad_, rad_));
        //cout << "b.channels = " << b.channels() << endl;         // for test

        split(meanA, vecA);
        vecP[i] = vecA[0].mul(vecI[0]) + vecA[1].mul(vecI[1]) + vecA[2].mul(vecI[2]) + meanB;
    }
    merge(vecP, imgOut);
}
