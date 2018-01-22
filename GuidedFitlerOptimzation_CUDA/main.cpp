#include <nppi.h>
#include "GuidedFilter.h"

using namespace std;
using namespace cv;

int main() {
    //std::cout << "Hello, World!" << std::endl;
    cudaDeviceProp cudaProp;
    int cudaId;
    cudaGetDevice(&cudaId);
    cudaGetDeviceProperties(&cudaProp, cudaId);
    cout << "Device ID: " << cudaId << endl;
    cout << "Concurrent Kernel Execution: " << cudaProp.concurrentKernels << endl;

    Mat imgI = imread("Barbara.jpg", IMREAD_COLOR);
    assert(!imgI.empty());
    const int row = imgI.rows;
    const int col = imgI.cols;
    imgI.convertTo(imgI, CV_32FC3, 1.0 / 255);
    const float* imgI_h = (const float*)imgI.data;
    imshow("Input", imgI);
    Mat imgP = imgI;
    cout << "Image Info: " << endl;
    cout << "Size: " << row << " * " << col << endl;
    cout << "Channels: " << imgI.channels() << endl << endl;
    Mat imgOut = Mat::zeros(Size(col, row), imgI.type());
    //GFilter gf(imgI.rows, imgI.cols);
    //gf.setParams(16, 0.01);   // Image Enhancement

    //boxFilter(imgI, imgOut, imgI.depth(), Size(16, 16));
    float* imgOut_h = (float *)imgOut.data;
    assert(imgI.isContinuous());
    int pSrcStepBytes = col * sizeof(float) * imgI.channels();
    //cout << "Src Step: " << pSrcStepBytes << endl;
    int pStepBytes;
    Npp32f* imgIn_d = nppiMalloc_32f_C3(col, row, &pStepBytes);
    NppStatus stateNpp = NPP_SUCCESS;
    cudaError_t stateCUDA = cudaSuccess;
    NppiSize sizeROI;
    sizeROI.width = col;
    sizeROI.height = row;
    //stateNpp = nppiCopy_32f_C3R(imgI_h, pSrcStepBytes, imgIn_d, pStepBytes, sizeROI);
    //assert(stateNpp == NPP_SUCCESS);
    // Copy image from host to device
    stateCUDA = cudaMemcpy2D(imgIn_d, pStepBytes, imgI_h, pSrcStepBytes, pSrcStepBytes, row, cudaMemcpyHostToDevice);
    assert(stateCUDA == cudaSuccess);
    //cout << imgOut(Range(0, 10), Range(90, 100)) << endl;
    //imshow("Temp", imgOut);
    Npp32f* imgOut_d = nppiMalloc_32f_C3(col, row, &pStepBytes);
    NppiSize oMaskSize = {16, 16};
    NppiPoint oAnchor = {oMaskSize.width/2, oMaskSize.height / 2};



    // boxFilter(imgI, imgOut, imgI.depth(), Size(16, 16));
    // gf.guidedfilterOpenCV(imgOut, imgI, imgI);
    //stateNpp = nppiFilterBoxBorder_32f_C3R(imgIn_d, pStepBytes, sizeROI, {0,0}, imgOut_d, pStepBytes, sizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    // begin test module
    // -------------------------- boxFilter OpenCV -------------------------- //
    // boxFilter on OpenCV
    // boxFilter(imgI, imgOut, imgI.depth(), Size(16, 16));
    // -------------------------- boxFilter OpenCV -------------------------- //

    // -------------------------- nppiFilterBox_32f_C3R -------------------------- //
    stateNpp = nppiFilterBoxBorder_32f_C3R(imgIn_d, pStepBytes, sizeROI, {0,0}, imgOut_d, pStepBytes, sizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    if (stateNpp != NPP_SUCCESS)
    {
        nppiFree(imgIn_d);
        nppiFree(imgOut_d);
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    // -------------------------- nppiFilterBox_32f_C3R -------------------------- //

    // -------------------------- guided filter OpenCV -------------------------- //
    //gf.guidedfilterOpenCV(imgOut, imgI, imgP);
    // normalize(imgOut, imgOut, 0, 1, CV_MINMAX);
    // -------------------------- guided filter OpenCV -------------------------- //
    chrono::steady_clock::time_point stop = chrono::steady_clock::now();
    chrono::duration<double> elapsedTime = static_cast<chrono::duration<double>>(stop - start);
    cout << "Elapsed Time: " << elapsedTime.count() * 1000.0 << "ms." << endl;
    stateCUDA = cudaMemcpy2D(imgOut_h, pSrcStepBytes, imgOut_d, pStepBytes, pStepBytes, row, cudaMemcpyDeviceToHost);
    assert(stateCUDA == cudaSuccess);
    cudaDeviceSynchronize();
    nppiFree(imgIn_d);
    nppiFree(imgOut_d);
    imshow("Result", imgOut);
    waitKey(0);

    return 0;
}
