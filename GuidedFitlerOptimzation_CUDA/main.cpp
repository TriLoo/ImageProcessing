#include <nppi.h>
#include "GuidedFilter.h"

using namespace std;
using namespace cv;

struct TestFloat4
{
    float x;
    float y;
    float z;
    float w;
};

int main() {
    //std::cout << "Hello, World!" << std::endl;
    cudaDeviceProp cudaProp;
    int cudaId, cudaVersion;
    cudaGetDevice(&cudaId);
    cudaGetDeviceProperties(&cudaProp, cudaId);
    cudaDriverGetVersion(&cudaVersion);
    cout << "Device ID: " << cudaId << endl;
    cout << "Concurrent Kernel Execution: " << cudaProp.concurrentKernels << endl;
    cout << "CUDA Version: " << cudaVersion << endl;
    cout << "GPU Asynchronize Engine Count: " << cudaProp.asyncEngineCount << endl;
    cout << "Registers per SM: " << cudaProp.regsPerMultiprocessor << endl;
    cout << "Count of SM:  " << cudaProp.multiProcessorCount << endl;
    cout << "Concurrent kernels: " << cudaProp.concurrentKernels << endl;
    cout << "Register per Block: " << cudaProp.regsPerBlock << endl;

    cout << "Test size of Float4." << endl;
    cout << sizeof(TestFloat4) << endl;         // return 16
    cout << "Test size of Float4." << endl;

    Mat imgI = imread("Barbara.jpg", IMREAD_COLOR);
    assert(!imgI.empty());
    const int row = imgI.rows;
    const int col = imgI.cols;
    imgI.convertTo(imgI, CV_32FC3, 1.0 / 255);
    imshow("Input", imgI);
    Mat imgP = imgI;
    cout << "Image Info: " << endl;
    cout << "Size: " << row << " * " << col << endl;
    cout << "Channels: " << imgI.channels() << endl << endl;
    Mat imgOut = Mat::zeros(Size(col, row), imgI.type());

    //GFilter gf(imgI.rows, imgI.cols);
    //gf.setParams(16, 0.01);   // Image Enhancement

    //boxFilter(imgI, imgOut, imgI.depth(), Size(16, 16));
    // gf.guidedfilterOpenCV(imgOut, imgI, imgI);
    //stateNpp = nppiFilterBoxBorder_32f_C3R(imgIn_d, pStepBytes, sizeROI, {0,0}, imgOut_d, pStepBytes, sizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    // begin test module
    // -------------------------- boxFilter OpenCV -------------------------- //
    // boxFilter on OpenCV
    // boxFilter(imgI, imgOut, imgI.depth(), Size(16, 16));
    // -------------------------- boxFilter OpenCV -------------------------- //

    // -------------------------- nppiFilterBox_32f_C3R -------------------------- //
     //gf.boxfilterNpp(imgOut, imgI);
    // -------------------------- nppiFilterBox_32f_C3R -------------------------- //

    // -------------------------- guided filter OpenCV -------------------------- //
    // gf.guidedfilterOpenCV(imgOut, imgI, imgP);
    // normalize(imgOut, imgOut, 0, 1, CV_MINMAX);
    // -------------------------- guided filter OpenCV -------------------------- //

    // -------------------------- Test Pinned Memory -------------------------- //
    float *inP = (float *)imgI.data;
    float *outP = (float *)imgOut.data;

    cudaError_t cudaState = cudaSuccess;
    cudaState = cudaHostRegister(inP, sizeof(float) * imgI.channels() * row * col, cudaHostRegisterDefault);
    assert(cudaState == cudaSuccess);
    cudaState = cudaHostRegister(outP, sizeof(float) * imgI.channels() * row * col, cudaHostRegisterDefault);
    assert(cudaState == cudaSuccess);

    float* in_d;
    size_t srcPatch;
    cudaState = cudaMallocPitch((void **)&in_d, &srcPatch, sizeof(float) * col * imgI.channels(), row);    // The width unit is byte
    assert(cudaState == cudaSuccess);
    cudaState = cudaMemcpy2DAsync(in_d, srcPatch, inP, col * sizeof(float) * imgI.channels(), col * sizeof(float) * imgI.channels(), row, cudaMemcpyHostToDevice);
    assert(cudaState == cudaSuccess);
    cudaState = cudaMemcpy2DAsync(outP, col * sizeof(float) * imgI.channels(), in_d, srcPatch, srcPatch, row, cudaMemcpyDeviceToHost);
    assert(cudaState == cudaSuccess);

    cudaState = cudaHostUnregister(inP);
    assert(cudaState == cudaSuccess);
    cudaState = cudaHostUnregister(outP);
    assert(cudaState == cudaSuccess);
    // -------------------------- Test Pinned Memory -------------------------- //
    chrono::steady_clock::time_point stop = chrono::steady_clock::now();
    chrono::duration<double> elapsedTime = static_cast<chrono::duration<double>>(stop - start);
    cout << "Total Elapsed Time: " << elapsedTime.count() * 1000.0 << "ms." << endl;
    imshow("Result", imgOut);
    waitKey(0);

    return 0;
}
