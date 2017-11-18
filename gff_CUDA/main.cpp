//#include <iostream>
//#include "BFilter.h"
//#include "TwoScale.h"
#include "GFilter.h"
#include "chrono"

#define TWOIMG
#undef TWOIMG

#define BFRAD 15
#define GFRAD 10
#define GFEPS 0.1

int main() {
    std::cout << "Hello, World!" << std::endl;

    //Mat imgInA = imread("lena.jpg", IMREAD_COLOR);
    Mat imgInA = imread("Visual.jpg", IMREAD_COLOR);
    //Mat imgInA = imread("Infrared.jpg", IMREAD_COLOR);
    try
    {
        if(!imgInA.data)
            throw runtime_error("Read Image failed.");
    }
    catch (runtime_error err)
    {
        cerr << err.what() << endl;
        return EXIT_FAILURE;
    }

    // convert input image to grayscale
    cvtColor(imgInA, imgInA, COLOR_RGB2GRAY);
    imshow("Input Image A", imgInA);

    imgInA.convertTo(imgInA, CV_32F, 1.0);
    assert(imgInA.isContinuous() == true);
    float *imgInA_P = (float *)imgInA.data;

    // output image
    Mat imgOutA = Mat::zeros(imgInA.size(), CV_32F);
    assert(imgOutA.isContinuous() == true);
    float *imgOutA_P = (float *)imgOutA.data;

    Mat imgOutB = Mat::zeros(imgInA.size(), CV_32F);
    assert(imgOutB.isContinuous() == true);
    float *imgOutB_P = (float *)imgOutB.data;

    // declares
    //BFilter bf;
    //TwoScale ts;
    GFilter gf(imgInA.cols, imgInA.rows);   // (wid, hei)

    // Test
    // time calculation
    chrono::steady_clock::time_point startPoint = chrono::steady_clock::now();

    // Test BFilter
    //bf.boxfilterTest(imgOutP, imgInP, imgIn.cols, imgIn.rows, BFRAD);
    //ts.boxfilterTest(imgOutA_P, imgInA_P, imgInA.cols, imgInA.rows, BFRAD);
    //ts.twoscaleTest(imgOutA_P, imgOutB_P, imgInA_P, imgInA.cols, imgInA.rows, BFRAD);
    gf.guidedfilterTest(imgOutA_P, imgInA_P, imgInA_P, imgInA.cols, imgInA.rows, GFRAD, GFEPS);

    chrono::steady_clock::time_point stopPoint = chrono::steady_clock::now();
    chrono::duration<double> elapsedTime = chrono::duration_cast<chrono::duration<double>>(stopPoint - startPoint);
    cout << "Test Time : " << elapsedTime.count() * 1000.0 << " ms." << endl;

    imgOutA.convertTo(imgOutA, CV_8UC1, 1.0);
    imshow("Output Image H", imgOutA);
#ifdef TWOIMG
    imgOutB.convertTo(imgOutB, CV_8UC1, 1.0);
    imshow("Output Image L", imgOutB);
#endif

    waitKey(0);

    return 0;
}
