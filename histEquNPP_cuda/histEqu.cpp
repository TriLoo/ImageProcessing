#include "histEqu.h"

int histEquNPP(Mat &imgOut, Mat &imgIn)
{
    const int rows = imgIn.rows;
    const int cols = imgIn.cols;

    // convert the image into gray scale

    //imgIn.convertTo(imgIn, CV_32F, 1.0);
    //imgOut.convertTo(imgOut, CV_32F, 1.0);
    cvtColor(imgIn, imgIn, COLOR_RGB2GRAY);

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

    waitKey(0);

    return EXIT_SUCCESS;
}
