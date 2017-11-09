#include <iostream>
#include "stdexcept"
#include "vector"
#include "cassert"
//#include "chrono"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cufft.h"

using namespace std;
using namespace cv;

#define SIGMA 120
#define BLOCKSIZE 16

__global__ void CreateGaussian_kernel(float *d_io, double sigma, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= wid || idy >= hei)
        return ;

    int offset = idy * wid + idx;

    float val;
    int halfWid = wid >> 1;
    int halfHei = hei >> 1;

    val = expf(-(powf(idx - halfWid, 2) + powf(idy - halfHei, 2)) / (2 * sigma * sigma));

    d_io[offset] = val;
}

void CreateGaussain(float *d_filterData, double sigma, int wid, int hei)
{
    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    blockPerGrid.x = (wid + BLOCKSIZE - 1) / BLOCKSIZE;
    blockPerGrid.y = (hei + BLOCKSIZE - 1) / BLOCKSIZE;
    CreateGaussian_kernel<<<blockPerGrid, threadPerBlock>>>(d_filterData, sigma, wid, hei);
    cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;
}

// c is scale number
__global__ void modulateAndNormalize_kernel(cufftComplex *d_out, cufftComplex *d_in, float c, int wid, int hei)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int fftWid = (wid >> 1) + 1;

    if(idx >= fftWid || idy >= hei)
        return ;

    //int fftWid = (wid >> 1) + 1;
    //int fftHei = (hei >> 1) + 1;

    const int offset = idy * fftWid + idx;

    cufftComplex a = d_out[offset];
    cufftComplex b = d_in[offset];

    float2 R = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};

    d_out[offset] = R;
    if(idx < 10 && idy < 10)
    	printf("R.x = %f \n", R.x);
}

void modulateAndNormalize(cufftComplex *d_out, cufftComplex *d_in, int wid, int hei)
{
    dim3 threadPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid;
    int fftWid = (wid >> 1) + 1;
    cout << "fftWid = " << fftWid << endl;
    blockPerGrid.x = (fftWid + BLOCKSIZE - 1) / BLOCKSIZE;
    blockPerGrid.y = (hei + BLOCKSIZE - 1) / BLOCKSIZE;

    modulateAndNormalize_kernel<<<blockPerGrid, threadPerBlock>>>(d_out, d_in, 1.0f / (float)(wid * hei), wid, hei);
    cout << "Scale in gaussian : " << 1.0f / (float)(wid * hei) << endl;
    cout << cudaGetErrorString(cudaPeekAtLastError()) << endl;
}

void GaussianFFT(Mat &out, Mat &in)
{
    const int wid = in.cols;
    const int hei = in.rows;

    const int halfWid = (wid >> 1) + 1;
    //const int halfHei = (hei >> 1) + 1;

    cudaError_t cudaState = cudaSuccess;
    cufftResult fftState = CUFFT_SUCCESS;
    float *d_filter, *d_in, *d_out;            // filter, input datas, output datas
    cudaState = cudaMalloc((void **)&d_filter, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_in, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_out, sizeof(float) * wid * hei);
    assert(cudaState == cudaSuccess);

    cufftComplex *d_dataSpec, *d_filterSpec;   // the results in fft frequency domain
    //cufftComplex *d_tempSpec;
    // See data layout for more details
    cudaState = cudaMalloc((void **)&d_dataSpec, sizeof(cufftComplex) * halfWid * hei);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMalloc((void **)&d_filterSpec, sizeof(cufftComplex) * halfWid * hei);
    assert(cudaState == cudaSuccess);
    //cudaState = cudaMalloc((void **)&d_tempSpec, sizeof(cufftComplex) * halfWid * hei);
    //assert(cudaState == cudaSuccess);

    float *imgIn, *imgOut;
    imgIn = (float *)in.data;
    imgOut = (float *)out.data;

    // copy data from host to device
    cudaState = cudaMemcpy(d_in, imgIn, sizeof(float) * wid * hei, cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);

    /*
    // for test
    for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < 10; j++)
			cout << imgIn[i * wid + j] << "; ";
		cout << endl;
	}
	*/

    // create gassian filter matrix
    CreateGaussain(d_filter, SIGMA, wid, hei);

    // for test
    /*
    float *h_testFilter = new float [wid * hei];
    cudaState = cudaMemcpy(h_testFilter, d_filter, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++)
    {
    	for(int j = 0; j < 10; j++)
    		cout << h_testFilter[i * wid + j] << "; ";
    	cout << endl;
    }

    delete [] h_testFilter;
     */

    // prepare FFT handler
    //cufftHandle fftFwd, fftInv;
    cufftHandle planFwd, planInv;
    cufftPlan2d(&planFwd, hei, wid, CUFFT_R2C);
    cufftPlan2d(&planInv, hei, wid, CUFFT_C2R);

    // do the forward fft
    fftState = cufftExecR2C(planFwd, d_in, d_dataSpec);
    assert(fftState == CUFFT_SUCCESS);
    fftState = cufftExecR2C(planFwd, d_filter, d_filterSpec);
    assert(fftState == CUFFT_SUCCESS);

    // do the modulate and normalized
    modulateAndNormalize(d_dataSpec, d_filterSpec, wid, hei);

    // do the inverse fft
    fftState = cufftExecC2R(planInv, d_dataSpec, d_out);




    // copy back the data from device to host
    cudaState = cudaMemcpy(imgOut, d_out, sizeof(float) * wid * hei, cudaMemcpyDeviceToHost);
    assert(cudaState == cudaSuccess);

    /*
    // for test
    for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < 10; j++)
			cout << imgOut[i * wid + j] << "; ";
		cout << endl;
	}
	*/

    cufftDestroy(planFwd);
    cufftDestroy(planInv);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_dataSpec);
    cudaFree(d_filter);
    cudaFree(d_filterSpec);
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
    try {
        if(!img.data)
        {
            throw runtime_error("Read Image failed.");
        }
    }
    catch(runtime_error err)
    {
        cerr << err.what() << endl;
        return -1;
    }

    //img = Mat::zeros(img.size(), CV_32F);

    if(img.isContinuous())
        cout << "Can use img.data." << endl;
    else
        cout << "Cannot use img.data" << endl;


    imshow("Input", img);

    img.convertTo(img, CV_32F, 1.0);
    Mat outImg = Mat::ones(img.size(), CV_32F);

    GaussianFFT(outImg, img);
    //assert(!outImg.data);
    // for test
    /*
    for(int i = 0; i < 10; i++)
    {
    	float *outData = (float *)outImg.ptr<uchar *>(i);
    	for(int j = 0; j < 10; j++)
    		cout << outData[j] << "; " << endl;
    	cout << endl;
    }
    */

    outImg.convertTo(outImg, CV_8UC1, 1.0);
    imshow("Result", outImg);

    waitKey(0);

    return 0;
}
