//#include <iostream>
#include "WMap.h"

#define LAPRAD 1
#define GAURAD 5
#define GAUSIG 0.01

int main() {
    clock_t start, stop;
    double elapsedTime = 0.0;

    Mat imgIn = imread("lena.jpg", IMREAD_GRAYSCALE);

    try
    {
        if(!imgIn.data)
            throw runtime_error("Read Image fialed ...");
    }
    catch(runtime_error err)
    {
        cerr << err.what() << endl;
        return -1;
    }

    int row = imgIn.rows;
    int col = imgIn.cols;

    cout << "Image Information : " << row << " * " << col << endl;

    imshow("Input", imgIn);

    imgIn.convertTo(imgIn, CV_32F, 1.0);
    float *imgInP = (float *)imgIn.data;
    Mat imgOut = Mat::zeros(Size(row, col), CV_32F);
    float *imgOutP = (float *)imgOut.data;

    WMap wm(col, row, LAPRAD, GAURAD);
    start = clock();
    //wm.laplacianAbsTest(imgOutP, imgInP, col, row, LAPRAD);
    //wm.gaussianTest(imgOutP, imgInP, col, row, GAURAD, GAUSIG);
    wm.saliencymapTest(imgOutP, imgInP, col, row, LAPRAD, GAURAD, GAUSIG);
    stop = clock();
    cout << "Laplacian filter time : " << 1000.0 * (stop - start) / CLOCKS_PER_SEC << " ms" << endl;

    imgOut.convertTo(imgOut, CV_8UC1, 1.0);
    imshow("Output", imgOut);

    waitKey(0);

    return 0;
}
