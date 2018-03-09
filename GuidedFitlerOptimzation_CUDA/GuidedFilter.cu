#include "GuidedFilter.h"
#include "helper_math.h"

#define BLOCKSIZE 32   // BLOCKSIZE * BLOCKSIZE threads per block

using namespace std;
using namespace cv;

texture<float4, cudaTextureType2D> rgbaTex;
texture<float4, cudaTextureType2D> rgbaTexI, rgbaTexP;
cudaArray *rgbaIn_d, *rgbaOut_d;

int iDiv(int a, int b)
{
    if (a % b == 0)
        return a / b;
    else
        return a / b + 1;
}

void imgShow(Mat img)
{
    imshow("Temp", img);
    waitKey(0);
}

GFilter::GFilter(int r, int c) : row_(r), col_(c), rad_(45), eps_(0.000001)
{
    cudaEventCreate(&startEvent_);
    cudaEventCreateWithFlags(&stopEvent_, cudaEventBlockingSync);

    // Caution: the pitch_ is also in bytes
    cudaCheckError(cudaMallocPitch(&tempA_, &pitch_, c * sizeof(float4), r));    // the width is in bytes
    cudaCheckError(cudaMallocPitch(&tempB_, &pitch_, c * sizeof(float4), r));
    cudaCheckError(cudaMallocPitch(&tempC_, &pitch_, c * sizeof(float4), r));
    cudaCheckError(cudaMallocPitch(&tempD_, &pitch_, c * sizeof(float4), r));

    cudaCheckError(cudaMallocPitch(&tempE_, &pitch_, c * sizeof(float4), r));
    cudaCheckError(cudaMallocPitch(&tempF_, &pitch_, c * sizeof(float4), r));

    cudaCheckError(cudaMallocPitch(&tempData_, &pitch_, c * sizeof(float4), r));
    pitch_ /= sizeof(float4);    // Unit: Changing from bytes to float4

    cudaCheckError(cudaStreamCreate(&stream1_));
    cudaCheckError(cudaStreamCreate(&stream2_));
}

GFilter::~GFilter()
{
    cudaEventDestroy(startEvent_);
    cudaEventDestroy(stopEvent_);

    cudaCheckError(cudaFree(tempA_));
    cudaCheckError(cudaFree(tempB_));
    cudaCheckError(cudaFree(tempC_));
    cudaCheckError(cudaFree(tempD_));

    cudaCheckError(cudaFree(tempE_));
    cudaCheckError(cudaFree(tempF_));

    cudaCheckError(cudaFree(tempData_));

    cudaCheckError(cudaStreamDestroy(stream1_));
    cudaCheckError(cudaStreamDestroy(stream2_));
}

// Test template kernel function
// Failed. Texture reference is a) the static file scope variable, b) don't satisfy the definition of non-type parameter for a template in C++
// Leading to the fact, it is very complicated to use texture as the parameters of template
/*
//typedef texture<float4, cudaTextureType2D> texTypeName;
//template <texTypeName texTemp>
template <texture<float4, cudaTextureType2D>& texTemp>
__global__ void d_boxfilter_texture(float4 *d_out, int row, int col, int rad)
{
}

//__global__ void d_boxfilter_texture<rgbaTex>(float4 *d_out, int row, int col, int rad);

template __global__ void d_boxfilter_texture<rgbaTexI>(float4 *d_out, int row, int col, int rad);
*/

// Kernel functions
// __device__
// do boxfilter
__global__ void
d_box_I_x(float4* d_out, int row, int col, int rad, size_t pitch)
{
    float scale = 1.0f / (float)((rad << 1) + 1.0f);
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < row)
    {
        float4 t = make_float4(0.0f);
        for (int x = -rad; x <= rad; ++x)
        {
            t += tex2D(rgbaTexI, x, y);
        }

        d_out[y * pitch] = t * scale;

        for (int x = 1; x < col; ++x)
        {
            t += tex2D(rgbaTexI, x + rad, y);
            t -= tex2D(rgbaTexI, x - rad - 1, y);
            d_out[y * pitch + x] = t * scale;
        }
    }
}
__global__ void
d_box_P_x(float4* d_out, int row, int col, int rad, size_t pitch)
{
    float scale = 1.0f / (float)((rad << 1) + 1.0f);
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < row)
    {
        float4 t = make_float4(0.0f);
        for (int x = -rad; x <= rad; ++x)
        {
            t += tex2D(rgbaTexP, x, y);
        }

        d_out[y * pitch] = t * scale;

        for (int x = 1; x < col; ++x)
        {
            t += tex2D(rgbaTexP, x + rad, y);
            t -= tex2D(rgbaTexP, x - rad - 1, y);
            d_out[y * pitch + x] = t * scale;
        }
    }
}

// CAUTION: The input is transposed in Texture Memory ! ! !
/*
__global__ void
testTexture(float4 *d_out, int row, int col)
{
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

    if( y < row )
    {
        float4 t = make_float4(0.0f);
        for (int x = 0; x < col; ++x)
        {
            t = tex2D(rgbaTex, x, y);

            d_out[x * row + y] = t;
        }
    }
}
*/

__global__ void
d_boxfilter_rgb_y(float4* d_out_, float4* d_in_, const int row, const int col, const int rad, size_t pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < col)
    {
        float4 *d_in = &d_in_[x];
        float4 *d_out = &d_out_[x];

        float scale = 1.0f / (float)((rad << 1) + 1.0f);

        float4 t = make_float4(0.0f);

        t = d_in[0] * rad;

        for (int y = 0; y < (rad + 1); y++)
        {
            t += d_in[y * pitch];
        }

        d_out[0] = t * scale;

        // do up edge
        for (int y = 1; y < rad + 1; y++)
        {
            t += d_in[(y + rad) * pitch];
            t -= d_in[0];
            d_out[y * pitch] = t * scale;
        }

        // do main loop
        for (int y = (1 + rad); y < (row - rad); y++)
        {
            t += d_in[(y + rad) * pitch];
            t -= d_in[(y - rad) * pitch];
            d_out[y * col] = t * scale;
        }

        // do right edge
        for (int y = row - rad; y < row; y++)
        {
            t += d_in[(row - 1) * pitch];
            t -= d_in[((y - rad) * pitch) - pitch];

            d_out [y * pitch] = t * scale;
        }
    }
}

// kernel function: do Square operation & boxfilter
// row direction
// d_out is 2D pitch memory
__global__ void
d_square_box_I_x(float4* d_out, int row, int col, int rad, size_t pitch)
{
    float scale = 1.0f / (float)((rad << 1) + 1.0f);
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < row)
    {
        float4 t = make_float4(0.0f);
        for (int x = -rad; x <= rad; ++x)
        {
            t += tex2D(rgbaTexI, x, y) * tex2D(rgbaTexI, x, y);
        }

        d_out[y * pitch] = t * scale;

        for (int x = 1; x < col; ++x)
        {
            t += tex2D(rgbaTexI, x + rad, y) * tex2D(rgbaTexI, x + rad, y);
            t -= tex2D(rgbaTexI, x - rad - 1, y) * tex2D(rgbaTexI, x - rad - 1, y);
            d_out[y * pitch + x] = t * scale;
        }
    }
}
__global__ void
d_square_box_Ip_x(float4* d_out, int row, int col, int rad, size_t pitch)
{
    float scale = 1.0f / (float) ((rad << 1) + 1.0f);
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < row) {
        float4 t = make_float4(0.0f);
        for (int x = -rad; x <= rad; ++x) {
            t += tex2D(rgbaTexI, x, y) * tex2D(rgbaTexP, x, y);
        }

        d_out[y * pitch] = t * scale;

        for (int x = 1; x < col; ++x) {
            t += tex2D(rgbaTexI, x + rad, y) * tex2D(rgbaTexP, x + rad, y);
            t -= tex2D(rgbaTexI, x - rad - 1, y) * tex2D(rgbaTexP, x - rad - 1, y);
            d_out[y * pitch + x] = t * scale;
        }
    }
}

// the d_in and d_out are both 2D pitch memory
__global__ void
d_square_box_y(float4* d_out_, float4* d_in_, const int row, const int col, const int rad, size_t pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < col)
    {
        float4 *d_in = &d_in_[x];
        float4 *d_out = &d_out_[x];

        float scale = 1.0f / (float)((rad << 1) + 1.0f);

        float4 t = make_float4(0.0f);

        t = d_in[0] * rad;

        for (int y = 0; y < (rad + 1); y++)
        {
            t += d_in[y * pitch] * d_in[y * pitch];
        }

        d_out[0] = t * scale;

        // do up edge
        for (int y = 1; y < rad + 1; y++)
        {
            t += d_in[(y + rad) * pitch] * d_in[(y + rad) * pitch];
            t -= d_in[0] * d_in[0];
            d_out[y * pitch] = t * scale;
        }

        // do main loop
        for (int y = (1 + rad); y < (row - rad); y++)
        {
            t += d_in[(y + rad) * pitch] * d_in[(y + rad) * pitch];
            t -= d_in[(y - rad) * pitch] * d_in[(y - rad) * pitch];
            d_out[y * pitch] = t * scale;
        }

        // do right edge
        for (int y = row - rad; y < row; y++)
        {
            t += d_in[(row - 1) * pitch] * d_in[(row - 1) * pitch];
            t -= d_in[((y - rad) * pitch) - col] * d_in[((y - rad) * pitch) - pitch];

            d_out [y * pitch] = t * scale;
        }
    }
}

// calculate the mean of a:
// varI = corrI - meanI .* meanI;
// covIp = corrIp - meanI .* meanP;
// a = corIp ./ (varI + eps);
// meanA = fmean(a);   this is implemented in the host function calling boxfilter, not a single kernel function
__global__ void calculateA(float4 *out, float4 *corrI, float4 *meanI, float4 *corrIp, float4 *meanP, double eps, int rows, int cols, size_t pitch)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= cols || idy >= rows)
        return ;

    int index = idx + idy * pitch;

    float4 valI = meanI[index];
    float4 varI = corrI[index] - valI * valI;
    float4 covIp = corrIp[index] - valI * meanP[index];
    out[index] = covIp / (varI + eps);
    //outA[index] = covIp / (varI + eps);
    // calculate B at the same function
    // outB[index] = meanP[index] - out[index] * valI;
}

// calculate b
__global__ void calculateB(float4 *out, float4 *inA, float4 *meanI, float4 *meanP, int rows, int cols, size_t pitch)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= cols || idy >= rows)
        return ;

    int index = idx + idy * pitch;

    out[index] = meanP[index] - inA[index] * meanI[index];
}

// do Covariance Calculation
// do Variance Calculation
// do Calculation of b
// Caution: the Out is also the Minuend(被减数)
__global__ void
MinusMultipleKernel(float4* Out, float4* inA, float4 *inB, int rows, int cols, size_t pitch)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= cols || idy >= rows)
        return ;

    int index = idy * pitch + idx;
    Out[index] -= inA[index] * inB[index];
}

// Calculate A
// Now the Out is also the dividend (被除数)
__global__ void
//CalculateAKernle(float4 *Out, float4* inA, float4 *inB, double eps, int rows, int cols)
CalculateAKernle(float4 *Out, float4* in, double eps, int rows, int cols, size_t pitch)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= cols || idy >= rows)
        return ;

    int index = idy * pitch + idx;
    Out[index] /= in[index] + eps;
}

// Calculate Final Value
// Caution: the Out is also the input of meanB
// inA: meanA, inB: the input image, or texture
__global__ void
//CalculateQ(float4 *Out, float4 *inA, float4 *inB, float4 *inC)
//CalculateQ(float4 *Out, float4 *inA, float4 *inB, int rows, int cols)
calculateQ(float4 *Out, float4 *in, int rows, int cols, size_t pitch)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= cols || idy >= rows)
        return ;

    int index = idy * pitch + idx;
    float4 tempVal = tex2D(rgbaTexI, idx, idy);
    Out[index] += in[index] * tempVal;
}

// begin define the functions in GFilter
void GFilter::initTexture(const cv::Mat &inI, const cv::Mat &inP)
{
    // convert input to RGBA
    if (inI.channels() == 3)
        cvtColor(inI, inI_, CV_BGR2BGRA);
    if (inP.channels() == 3)
        cvtColor(inP, inP_, CV_BGR2BGRA);

    if (inI_.type() != CV_32FC4)
        inI_.convertTo(inI_, CV_32FC4, 1.0 / 255);
    if (inP_.type() != CV_32FC4)
        inP_.convertTo(inP_, CV_32FC4, 1.0 / 255);

    assert(inI_.channels() == 4);
    assert(inP_.channels() == 4);

    float *inI_P = (float*)inI_.data;
    float *inP_P = (float*)inP_.data;
    // use Asynchronous copy, from host memory to 2D Pitch memory
    // Register the host memory as pinned memory
    cudaCheckError(cudaHostRegister(inI_P, sizeof(float4) * row_ * col_, cudaHostRegisterDefault));
    cudaCheckError(cudaHostRegister(inP_P, sizeof(float4) * row_ * col_, cudaHostRegisterDefault));

    // Copy data from CPU to GPU
    cudaCheckError(cudaMemcpyAsync(tempF_, inI_P,sizeof(float4) * row_ * col_, cudaMemcpyHostToDevice, stream1_));
    cudaCheckError(cudaMemcpyAsync(tempC_, inP_P,sizeof(float4) * row_ * col_, cudaMemcpyHostToDevice, stream2_));

    // bind the above two memories to texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaCheckError(cudaBindTexture2D(0, rgbaTexI, tempF_, channelDesc, col_, row_, pitch_ * sizeof(float4)));
    cudaCheckError(cudaBindTexture2D(0, rgbaTexP, tempC_, channelDesc, col_, row_, pitch_ * sizeof(float4)));
}

void GFilter::initTexture(float* data)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* tempH = new float [row_ * col_ * 4];          // 4 : the unmber of float in float4
    float* tempSrc = tempH;
    float* tempD = data;
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
    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaCheckError(cudaMallocArray(&rgbaIn_d, &channelDesc, col_, row_));

    cudaEventRecord(start);
    cudaCheckError(cudaMemcpyToArray(rgbaIn_d, 0, 0, tempSrc, size * sizeof(float4), cudaMemcpyHostToDevice));
    // Until here, It is correct ! ! !

    // bind array to texture
    cudaCheckError(cudaBindTextureToArray(rgbaTex, rgbaIn_d, channelDesc));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0.0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Copy Data + Bind Texture: " << elapsedTime << " ms." << endl;

    delete [] tempSrc;
}

void releaseTexture()
{
    cudaUnbindTexture(rgbaTexI);
    cudaUnbindTexture(rgbaTexP);
    //cudaCheckError(cudaFreeArray(rgbaOut_d));
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

// calculate boxfilter stored in tempC_
void GFilter::boxfilterTempC(float4* imgIO)
{
    // Measure the time used to calculate boxfilter of tempC_, binded to rgbaTexP;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // use texture for horizontal pass
    //dim3 blockPerGrid = ;
    d_box_P_x<<<iDiv(row_, BLOCKSIZE), BLOCKSIZE, 0>>>(tempData_, row_, col_, rad_, pitch_);   // use row_ / BLOCKSIZE, because the input is transposed.
    //d_boxfilter_rgb_x<<<iDiv(row_, BLOCKSIZE), BLOCKSIZE, 0>>>(outDataD, row_, col_, rad);
    //testTexture<<<row_ / BLOCKSIZE, BLOCKSIZE, 0>>>(outDataD, row_, col_);     // The Result is transposed of input matrix
    d_boxfilter_rgb_y<<<iDiv(col_, BLOCKSIZE), BLOCKSIZE, 0>>>(imgIO, tempData_, row_, col_, rad_, pitch_);
    //cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0.0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Boxfilter on GPU(no data transfer: " << elapsedTime << " ms." << endl;
}
// calculate meanI
void GFilter::boxfilterImgI(float4 *imgIO)
{
    // Measure the time used to calculate boxfilter of tempC_, binded to rgbaTexP;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // use texture for horizontal pass
    //dim3 blockPerGrid = ;
    d_box_I_x<<<iDiv(row_, BLOCKSIZE), BLOCKSIZE, 0>>>(tempData_, row_, col_, rad_, pitch_);   // use row_ / BLOCKSIZE, because the input is transposed.
    //d_boxfilter_rgb_x<<<iDiv(row_, BLOCKSIZE), BLOCKSIZE, 0>>>(outDataD, row_, col_, rad);
    //testTexture<<<row_ / BLOCKSIZE, BLOCKSIZE, 0>>>(outDataD, row_, col_);     // The Result is transposed of input matrix
    d_boxfilter_rgb_y<<<iDiv(col_, BLOCKSIZE), BLOCKSIZE, 0>>>(imgIO, tempData_, row_, col_, rad_, pitch_);
    //cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0.0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Boxfilter on GPU(no data transfer: " << elapsedTime << " ms." << endl;
}

// calculate corrI & corrIp
void GFilter::boxfilterMultiple(float4 *corrI, float4 *corrIp)
{
    // Measure the time used to calculate boxfilter of tempC_, binded to rgbaTexP;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockPerGrid = iDiv(row_, BLOCKSIZE);
    dim3 threadPerBlock = BLOCKSIZE;

    cudaEventRecord(start);
    // use texture for horizontal pass
    // calculate corrI
    d_square_box_I_x<<<blockPerGrid, threadPerBlock, 0>>>(tempData_, row_, col_, rad_, pitch_);   // use row_ / BLOCKSIZE, because the input is transposed.
    //testTexture<<<row_ / BLOCKSIZE, BLOCKSIZE, 0>>>(outDataD, row_, col_);     // The Result is transposed of input matrix
    d_square_box_y<<<blockPerGrid, BLOCKSIZE, 0>>>(corrI, tempData_, row_, col_, rad_, pitch_);
    //cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    // calculate corrIp
    d_square_box_Ip_x<<<blockPerGrid, threadPerBlock, 0>>>(tempData_, row_, col_, rad_, pitch_);   // use row_ / BLOCKSIZE, because the input is transposed.
    //d_boxfilter_rgb_x<<<iDiv(row_, BLOCKSIZE), BLOCKSIZE, 0>>>(outDataD, row_, col_, rad);
    d_square_box_y<<<blockPerGrid, BLOCKSIZE, 0>>>(corrIp, tempData_, row_, col_, rad_, pitch_);
    //cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0.0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Boxfilter on GPU(no data transfer: " << elapsedTime << " ms." << endl;
}

void GFilter::boxfilterTest(cv::Mat &imgOut, const cv::Mat &imgIn, int rad)
{
}
/*
void GFilter::boxfilterTest(cv::Mat &imgOut, const cv::Mat &imgIn, int rad)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *dataInP = (float *)imgIn.data;
    float *dataOutP = (float *)imgOut.data;

    float4 *tempData, *outDataD;
    float *tempDataH = new float [row_ * col_ * sizeof(float4)];
    //cudaChannelFormatDesc channels = cudaCreateChannelDesc<float4>();
    cudaCheckError(cudaMalloc((void **)&tempData, sizeof(float4) * col_ * row_));
    cudaCheckError(cudaMalloc((void **)&outDataD, sizeof(float4) * col_ * row_));

    initTexture(dataInP);
    cudaEventRecord(start);
    // use texture for horizontal pass
    //dim3 blockPerGrid = ;
    d_boxfilter_rgb_x<<<iDiv(row_, BLOCKSIZE), BLOCKSIZE, 0>>>(tempData, row_, col_, rad);   // use row_ / BLOCKSIZE, because the input is transposed.
    //d_boxfilter_rgb_x<<<iDiv(row_, BLOCKSIZE), BLOCKSIZE, 0>>>(outDataD, row_, col_, rad);
    //testTexture<<<row_ / BLOCKSIZE, BLOCKSIZE, 0>>>(outDataD, row_, col_);     // The Result is transposed of input matrix
    d_boxfilter_rgb_y<<<iDiv(col_, BLOCKSIZE), BLOCKSIZE, 0>>>(outDataD, tempData, row_, col_, rad);
    //cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0.0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Boxfilter on GPU(no data transfer: " << elapsedTime << " ms." << endl;

    cudaCheckError(cudaMemcpy(tempDataH, outDataD, sizeof(float4) * row_ * col_, cudaMemcpyDeviceToHost));
    //cudaCheckError(cudaMemcpy(tempDataH, tempData, sizeof(float4) * row_ * col_, cudaMemcpyDeviceToHost));
    //cudaCheckError(cudaMemcpyFromArray(tempDataH, rgbaIn_d, 0, 0, sizeof(float4) * row_ * col_, cudaMemcpyDeviceToHost));    // CORRECT ! ! !

    restoreFromFloat4(dataOutP, tempDataH);

    delete [] tempDataH;
    releaseTexture();
}
*/

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
void GFilter::guidedfilterSingle(cv::Mat &imgOut, const cv::Mat &imgInI)
{
    // Note: Change the pitch_ into pitch which is in bytes
    //       Use this value to index the elements in the 2D pitch memory
    //size_t pitch = pitch_ / sizeof(float4);
    // do  boxfilter
}

// 输入图像是不同的  e.g. imgInI != imgInP
void GFilter::guidedfilterDouble(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    // Initialize
    //setParams(15, 0.01);   // use the default values of radius and eps
    setParams();
    initTexture(imgInI, imgInP);

    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid(iDiv(col_, BLOCKSIZE), iDiv(row_, BLOCKSIZE));     // dim3(width, height) ! ! !

    cudaEventRecord(startEvent_);
    // Step 1
    boxfilterMultiple(tempA_, tempD_);
    boxfilterImgI(tempB_);
    boxfilterTempC(tempE_);

    // Step 2: calculate A and meanA
    calculateA<<<blockPerGrid, threadPerBlock, 0>>>(tempC_, tempA_, tempB_, tempD_, tempE_, eps_, row_, col_, pitch_);
    // synchronize the device before going on to use the texture P
    cudaDeviceSynchronize();
    // get the mean of A
    boxfilterTempC(tempD_);

    // Step 3: calculate B and meanB
    calculateB<<<blockPerGrid, threadPerBlock, 0>>>(tempC_, tempD_, tempB_, tempE_, row_, col_, pitch_);
    boxfilterTempC(tempA_);

    // Step 4: calculate the output: q = meanA .* I + meanB
    calculateQ<<<blockPerGrid, threadPerBlock, 0>>>(tempA_, tempD_, row_, col_, pitch_);

    // copy back image from device to host
    if (imgOut.empty())
        imgOut = Mat::zeros(imgInI.size(), CV_32FC4);
    if (imgOut.channels() == 3)
        cvtColor(imgOut, imgOut, CV_BGR2BGRA);
    if (imgOut.type() != CV_32FC4)
        imgOut.convertTo(imgOut, CV_32FC4, 1.0/255);

    // sizeof: return the bytes of the object representation of type
    float *outP = (float *)imgOut.data;
    pitch_ *= sizeof(float4);
    cudaCheckError(cudaMemcpy2D(outP, sizeof(float4) * col_, tempA_, pitch_, col_ * sizeof(float4), row_, cudaMemcpyDeviceToHost));

    cudaEventRecord(stopEvent_);
    cudaEventSynchronize(stopEvent_);
    float elapsedTime = 0.0;
    cudaEventElapsedTime(&elapsedTime, startEvent_, stopEvent_);
    cout << "Exclude Texture Initialization: " << elapsedTime << " ms." << endl;
}


void GFilter::guidedfilter(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    assert(imgInP.channels() == 3 && imgInI.channels() == 3);
    //const float *imgA = (float *)imgInI.data;
    //const float *imgB = (float *)imgInP.data;
    equal_to<const float*> T;
    if (T((float *)imgInI.data, (float*)imgInP.data))
        guidedfilterSingle(imgOut, imgInI);
    else
        guidedfilterDouble(imgOut, imgInI, imgInP);
}

// Contrast Experiments: Guided Filter based on OpenCV
void GFilter::guidedfilterOpenCV(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    assert(imgInP.channels() == 3);
    if (rad_ == 0)
        setParams(45, 10^(-6));    // Image Enhancement  or Blur

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
//#pragma unloop
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
