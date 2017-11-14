#include "histEqu.h"

#define cudaCheckErrors(err) __cudaCheckErrors(err, __FILE__, __LINE__)

inline void __cudaCheckErrors(cudaError_t err, const char *file, const int line)
{
    if(err != cudaSuccess)
        cout << "Error :" << cudaGetErrorString(err) << " in " << file << " file. At " << line << " line." << endl;
}

// calculate histgram of gray-scale image
int histEqu::histEquNpp(Mat &imgOut, Mat &imgIn)
{
    const int rows = imgIn.rows;
    const int cols = imgIn.cols;
    const int imgSize = rows * cols;

    NppStatus nppState = NPP_SUCCESS;

    // convert the image into gray scale
    cvtColor(imgIn, imgIn, COLOR_RGB2GRAY);
    cvtColor(imgOut, imgOut, COLOR_RGB2GRAY);

    imshow("Image Input", imgIn);

    // assert the memory space is continuous
    try
    {
        if(!imgIn.isContinuous() || !imgOut.isContinuous())
            throw runtime_error("Memory space is not continuos.");
    }
    catch(runtime_error err)
    {
        cerr << err.what() << endl;
        return EXIT_FAILURE;
    }
    cout << "The memory space is continus." << endl;

    // memory by NPP
    Npp8u *pSrc;
    int pStep;              // output
    pSrc = nppiMalloc_8u_C1(cols, rows, &pStep);
    cout << "Pitch on CUDA (Bytes) : " << pStep << endl;

    // copy data from host to device
    //cudaCheckErrors(cudaMemcpy(pSrc, imgIn.data, sizeof(uchar) * rows * cols, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy2D(pSrc, pStep, imgIn.data, sizeof(uchar) * cols, sizeof(uchar) * cols, rows, cudaMemcpyHostToDevice));

    // declare ROI
    NppiSize oSizeROI = {cols, rows};

    // get the min & max value on the input image
    Npp8u *pMin, *pMax;
    // get the scratch buffer size
    int nBufferSize;
    nppiMinMaxGetBufferHostSize_8u_C1R(oSizeROI, &nBufferSize);
    // Allocate the scratch buffer
    Npp8u *pDeviceBuffer;
    cudaCheckErrors(cudaMalloc((void **)&pDeviceBuffer, nBufferSize));

    // calculate the min & max value
    cudaCheckErrors(cudaMalloc((void **)&pMin, sizeof(Npp8u) * 1));
    cudaCheckErrors(cudaMalloc((void **)&pMax, sizeof(Npp8u) * 1));

    nppState = nppiMinMax_8u_C1R(pSrc, pStep, oSizeROI, pMin, pMax, pDeviceBuffer);
    assert(nppState == NPP_SUCCESS);
    //nppState = nppiMinMax_8u_C1R(pSrc, pStep, {cols, rows}, );

    // copy min & max value back to host
    uchar h_min, h_max;  // if want to output the result, should use uchar, then static_cast to 'int' !
    cudaCheckErrors(cudaMemcpy(&h_min, pMin, sizeof(uchar) * 1, cudaMemcpyDeviceToHost));
    cudaCheckErrors(cudaMemcpy(&h_max, pMax, sizeof(uchar) * 1, cudaMemcpyDeviceToHost));

    cout << "Min value of input image is : " << static_cast<int>(h_min) << endl;
    cout << "Max value of input image is : " << static_cast<int>(h_max) << endl;

    // do hist calculation
    const int binCount = 255;
    const int levelCount = binCount + 1;   // is the

    // prepare the memory
    int *h_hist = new int [binCount];
    int *d_hist;
    cudaCheckErrors(cudaMalloc((void **)&d_hist, sizeof(int) * binCount));

    nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, levelCount, &nBufferSize);
    cudaCheckErrors(cudaMalloc((void **)&pDeviceBuffer, sizeof(Npp8u) * nBufferSize));

    // calculate the hist
    nppiHistogramEven_8u_C1R(pSrc, pStep, oSizeROI, d_hist, levelCount, 0, binCount, pDeviceBuffer);

    // copy hist back from device to host
    cudaCheckErrors(cudaMemcpy(h_hist, d_hist, sizeof(int) * binCount, cudaMemcpyDeviceToHost));

    //for(auto beg = begin(h_hist); beg != end(h_hist); beg++)
        //cout << *beg << endl;
    for(int i = 0; i < binCount; i++)
        cout <<"i = " << i << " : " << h_hist[i] << "; " << endl;


    // get the hist-equalized result
    // create LUT
    int *h_lut = new int[levelCount];
    int totalSum = 0;          // total pixel numbers
    {
        int *temp_hist = h_hist;
        for(; temp_hist < h_hist + binCount; ++temp_hist)
            totalSum += *temp_hist;

        cout << "Total pixel numbers : " << totalSum << endl;
        totalSum = totalSum == 0 ? 1 : totalSum;

        float lutScale = 1.0f / totalSum * 0xFF;   // 0xFF : 255
        cout << "lut Scale = " << lutScale << endl;

        int *pLUT = h_lut;
        int tempSum = 0;

        for (temp_hist = h_hist; temp_hist < h_hist + binCount; ++temp_hist)
        {
            *pLUT = static_cast<int>(lutScale * tempSum + 0.5f);
            pLUT++;
            tempSum += *temp_hist;
        }

        h_lut[binCount] = 0xFF;
    }

    for(int i = 0; i < levelCount; i++)
        cout << "h_lut =  "  << h_lut[i] << endl;

    Npp32s * d_lut;
    cudaCheckErrors(cudaMalloc((void **)&d_lut, sizeof(Npp32s) * levelCount));
    cudaCheckErrors(cudaMemcpy(d_lut, h_lut, sizeof(Npp32s) * levelCount, cudaMemcpyHostToDevice));

    // create level
    int *h_level = new int [levelCount];
    nppiEvenLevelsHost_32s(h_level, levelCount, 0, binCount);
    Npp32s *d_level;
    cudaCheckErrors(cudaMalloc((void **)&d_level, sizeof(Npp32s) * levelCount));
    cudaCheckErrors(cudaMemcpy(d_level, h_level, sizeof(Npp32s) * levelCount, cudaMemcpyHostToDevice));

    // result position
    Npp8u *pDst;
    int pDstStep;
    pDst = nppiMalloc_8u_C1(cols, rows, &pDstStep);

    nppState = nppiLUT_8u_C1R(pSrc, pStep, pDst, pDstStep, oSizeROI, d_lut, d_level, levelCount);
    assert(nppState == NPP_SUCCESS);

    // copy image back from device to host
    //cout << "Src pitch : " << sizeof(uchar) * cols << endl;
    //cout << "Dst pitch : " << pDstStep << endl;
    //cout << "Dst width : " << pDstStep << endl;
    //uchar *resultImg = new uchar[cols * rows];
    cudaCheckErrors(cudaMemcpy2D(imgOut.data, sizeof(uchar) * cols, pDst, static_cast<size_t>(pDstStep), cols * sizeof(uchar), rows, cudaMemcpyDeviceToHost));

    imshow("Output", imgOut);
    waitKey(0);

    //delete [] resultImg;
    delete [] h_hist;
    delete [] h_lut;
    delete [] h_level;
    cudaFree(d_hist);
    cudaFree(pSrc);
    //cudaFree(pDst);
    cudaFree(pMin);
    cudaFree(pMax);
    cudaFree(pDeviceBuffer);
    cudaFree(d_lut);
    cudaFree(d_level);

    return EXIT_SUCCESS;
}
