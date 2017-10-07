#include <iostream>
#include <nppi.h>
#include "fstream"
#include "vector"
#include "ctime"
#include "stdexcept"
#include "cassert"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "npp.h"
#include "nppi.h"
#include "FreeImage.h"

using namespace std;
using namespace cv;

#pragma comment(lib, "FreeImage.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "nppi.lib")

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *file, int line)
{
    if(err != cudaSuccess)
    {
        printf("%s(%i) : CUDA Runtime API error %d : %s .\n", file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

FIBITMAP* LoadImg(const char *sf)
{
    FREE_IMAGE_FORMAT nFif;

    if((nFif = FreeImage_GetFileType(sf, 0)) == FIF_UNKNOWN)
    {
        if((nFif = FreeImage_GetFIFFromFilename(sf)) == FIF_UNKNOWN)
            return nullptr;
    }

    if(!FreeImage_FIFSupportsReading(nFif))
        return nullptr;

    return FreeImage_Load(nFif, sf);
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);

    try
    {
        if(!img.data)
            throw runtime_error("Read image failed ...");
    }
    catch(runtime_error err)
    {
        cerr << err.what() << endl;
        return 1;
    }

    imshow("Input", img);
    img.convertTo(img, CV_32F, 1.0);
    int row = img.rows;
    int col = img.cols;

    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    cout << "GPU Device Count : " << deviceCount << endl;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cout << "cudaSetDevice GPU " << 0 << " = " << deviceProp.name << endl;
    //cout << "max Texture 2D : " << deviceProp.maxTexture2D << endl;

    cudaError_t cudaState = cudaSuccess;
    NppStatus nppState = NPP_SUCCESS;

    bool fiRet;

    FIBITMAP *pSrcBmp, *pDstBmp;
    unsigned char *pSrcData, *pDstData;
    Npp8u *pSrcDataCUDA, *pDstDataCUDA;

    NppiSize oSrcSize, oDstSize;
    NppiRect oSrcROI, oDstROI;

    int nImgBpp, nSrcPitch, nDstPitch, nSrcPitchCUDA, nDstPitchCUDA;
    //double aBoundingBox[2][2];
    //double nAngle;

    // load Image
    pSrcBmp = LoadImg("lena.jpg");
    assert(pSrcBmp != NULL);

    nImgBpp = (FreeImage_GetBPP(pSrcBmp));
    cout << "The depth of input image : " << nImgBpp << endl;

    pSrcData = FreeImage_GetBits(pSrcBmp);
    //cout << "PSrcData = " << pSrcData << endl;

    oSrcSize.width = (int)FreeImage_GetWidth(pSrcBmp);
    oSrcSize.height = (int)FreeImage_GetHeight(pSrcBmp);
    cout << "Src Size : " << oSrcSize.width << " * " << oSrcSize.height << endl;
    nSrcPitch = (int)FreeImage_GetPitch(pSrcBmp);
    cout << "Src Pitch : " << nSrcPitch << endl;

    oSrcROI.x = oSrcROI.y = 0;
    oSrcROI.width = oSrcSize.width;
    oSrcROI.height = oSrcSize.height;

    checkCudaErrors(cudaSetDevice(0));
    /*
    oDstSize.width = (int)FreeImage_GetWidth(pSrcBmp);
    oDstSize.height = (int)FreeImage_GetHeight(pSrcBmp);
    nDstPitch = (int)FreeImage_GetPitch(pSrcBmp);
    */

    // 分配显存
    pSrcDataCUDA = nppiMalloc_8u_C3(oSrcSize.width, oSrcSize.height, &nSrcPitchCUDA);
    cout << "Src Pitch On CUDA : " << nSrcPitchCUDA << endl;
    assert(pSrcDataCUDA != NULL);

    // 将原图传入显存
    checkCudaErrors(cudaMemcpy2D(pSrcDataCUDA, nSrcPitchCUDA, pSrcData, nSrcPitch, oSrcSize.width * nImgBpp, oSrcSize.height, cudaMemcpyHostToDevice));
    //cudaMemcpy2D(pSrcDataCUDA, nSrcPitchCUDA, pSrcData, nSrcPitch, oSrcSize.width * nImgBpp, oSrcSize.height, cudaMemcpyHostToDevice);
    // 建立目标图
    pDstBmp = FreeImage_Allocate(oDstSize.width, oDstSize.height, nImgBpp);
    assert(pDstBmp != NULL);
    pDstData = FreeImage_GetBits(pDstBmp);

    // 分配输出显存
    oDstSize.width = oSrcSize.width;
    oDstSize.height = oSrcSize.height;
    nDstPitch = (int)FreeImage_GetPitch(pDstBmp);

    oDstROI.x = oDstROI.y = 0;
    oDstROI.width = oDstSize.width;
    oDstROI.height = oDstSize.height;

    pDstDataCUDA  = nppiMalloc_8u_C3(oDstSize.width, oDstSize.height, &nDstPitchCUDA);
    assert(pDstDataCUDA != NULL);

    checkCudaErrors(cudaMemset2D(pDstDataCUDA, nDstPitchCUDA, 0,oDstSize.width * nImgBpp, oDstSize.height));

    // boxfilter处理
    //nppiFilterBoxBorder_8u_C3R()
    NppiSize oMaskSize = {5, 5};
    NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};
    nppState = nppiFilterBoxBorder_8u_C3R(pSrcDataCUDA, nSrcPitchCUDA, oDstSize, {0, 0}, pDstDataCUDA, nDstPitchCUDA, oDstSize, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    assert(nppState == NPP_NO_ERROR);


    // copy result from device back to host
    checkCudaErrors(cudaMemcpy2D(pDstData, nDstPitch, pDstDataCUDA, nDstPitchCUDA, oDstSize.width * nImgBpp, oDstSize.height, cudaMemcpyDeviceToHost));

    // save image
    fiRet = FreeImage_Save(FIF_BMP, pDstBmp, "ret.bmp");
    assert(fiRet);

    nppiFree(pSrcDataCUDA);
    nppiFree(pDstDataCUDA);

    FreeImage_Unload(pSrcBmp);
    FreeImage_Unload(pDstBmp);

    return 0;
}
