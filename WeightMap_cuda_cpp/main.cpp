//#include <iostream>
#include "WMap.h"

#define LAPRAD 1
#define GAURAD 5
#define GAUSIG 0.01
#define GUIRAD 10
#define GUIEPS 0.1

int main() {
    clock_t start, stop;
    double elapsedTime = 0.0;

    Mat imgInA = imread("source20_1.tif", IMREAD_GRAYSCALE);
    Mat imgInB = imread("source20_2.tif", IMREAD_GRAYSCALE);

    try
    {
        if(!imgInA.data || !imgInB.data)
            throw runtime_error("Read Image fialed ...");
    }
    catch(runtime_error err)
    {
        cerr << err.what() << endl;
        return -1;
    }

    int row = imgInA.rows;
    int col = imgInA.cols;

    cout << "Image Information : " << row << " * " << col << endl;

    imshow("Input A", imgInA);
    imshow("Input B", imgInB);

    imgInA.convertTo(imgInA, CV_32F, 1.0);
    imgInB.convertTo(imgInB, CV_32F, 1.0);

    float *imgA_P = (float *)(imgInA.data);
    float *imgB_P = (float *)(imgInB.data);

    Mat imgOutA = Mat::zeros(Size(col, row), CV_32F);
    Mat imgOutB = Mat::zeros(Size(col, row), CV_32F);

    float *imgOutA_P = (float *)imgOutA.data;
    float *imgOutB_P = (float *)imgOutB.data;

    WMap wm(col, row, LAPRAD, GAURAD);
    start = clock();
    wm.weightedmapTest(imgOutA_P, imgOutB_P, imgA_P, imgB_P, col, row, LAPRAD, GAURAD, GAUSIG, GUIRAD, GUIEPS);
    stop = clock();
    cout << "Weighted Map Based on GPU : " << 1000.0 * (stop - start) / CLOCKS_PER_SEC << " ms" << endl;

    /*
    for(int i = 0; i < 10; ++i)
    {
        for(int j = 244; j < 256; ++j)
            cout << imgOutA_P[i + j * col]  << ", ";

        cout << endl;
    }
    */

    //imgOutA.convertTo(imgOutA, CV_8UC1, 1.0);
    //imgOutB.convertTo(imgOutB, CV_8UC1, 1.0);
    imgOutA.convertTo(imgOutA, CV_8UC1, 255.0);
    imgOutB.convertTo(imgOutB, CV_8UC1, 255.0);

    imshow("Output A", imgOutA);
    imshow("Output B", imgOutB);

    waitKey(0);

    /*
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
    //GFilter gf(col, row);
    start = clock();
    //wm.laplacianAbsTest(imgOutP, imgInP, col, row, LAPRAD);
    //wm.gaussianTest(imgOutP, imgInP, col, row, GAURAD, GAUSIG);
    wm.saliencymapTest(imgOutP, imgInP, col, row, LAPRAD, GAURAD, GAUSIG);
    //gf.guidedfilterTest(imgOutP, imgInP, imgInP, col, row, 45, 0.3);
    stop = clock();
    cout << "Laplacian filter time : " << 1000.0 * (stop - start) / CLOCKS_PER_SEC << " ms" << endl;

    imgOut.convertTo(imgOut, CV_8UC1, 1.0);
    imshow("Output", imgOut);

    waitKey(0);
    */

    return 0;
}
