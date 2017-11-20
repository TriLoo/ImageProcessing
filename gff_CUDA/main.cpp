//#include <iostream>
//#include "BFilter.h"
//#include "TwoScale.h"
//#include "GFilter.h"
#include "WeightedMap.h"
#include "chrono"

#define TWOIMG
//#undef TWOIMG

// BoxFilter Radius
#define BFRAD 15
// Laplacian
#define LAPRAD 1
// Gaussian
#define GAURAD 5
#define GAUSIG 0.01
// Guided Filter
#define GFRAD 10
#define GFEPS 0.1

int main() {
    std::cout << "Hello, World!" << std::endl;

    //Mat imgInA = imread("lena.jpg", IMREAD_COLOR);
    Mat imgInA = imread("Visual.jpg", IMREAD_COLOR);
    Mat imgInB = imread("Infrared.jpg", IMREAD_COLOR);
    //Mat imgInA = imread("Infrared.jpg", IMREAD_COLOR);
    try
    {
        if(!imgInA.data || !imgInB.data)
            throw runtime_error("Read Image failed.");
    }
    catch (runtime_error err)
    {
        cerr << err.what() << endl;
        return EXIT_FAILURE;
    }

    // convert input image to grayscale
    cvtColor(imgInA, imgInA, COLOR_RGB2GRAY);
    cvtColor(imgInB, imgInB, COLOR_RGB2GRAY);
    imshow("Input Image A", imgInA);
    imshow("Input Image B", imgInB);

    imgInA.convertTo(imgInA, CV_32F, 1.0);
    assert(imgInA.isContinuous() == true);
    float *imgInA_P = (float *)imgInA.data;
    imgInB.convertTo(imgInB, CV_32F, 1.0);
    assert(imgInB.isContinuous() == true);
    float *imgInB_P = (float *)imgInB.data;

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
    //GFilter gf(imgInA.cols, imgInA.rows);   // (wid, hei)
    WMap wm(imgInA.cols, imgInA.rows, LAPRAD, GFRAD);

    // Test
    // time calculation
    chrono::steady_clock::time_point startPoint = chrono::steady_clock::now();

    // Test BFilter
    //bf.boxfilterTest(imgOutP, imgInP, imgIn.cols, imgIn.rows, BFRAD);
    //ts.boxfilterTest(imgOutA_P, imgInA_P, imgInA.cols, imgInA.rows, BFRAD);
    //ts.twoscaleTest(imgOutA_P, imgOutB_P, imgInA_P, imgInA.cols, imgInA.rows, BFRAD);
    //gf.guidedfilterTest(imgOutA_P, imgInA_P, imgInA_P, imgInA.cols, imgInA.rows, GFRAD, GFEPS);
    //wm.saliencymapTest(imgOutA_P, imgInA_P, imgInA.cols, imgInA.rows, LAPRAD, GAURAD, GAUSIG);
    wm.weightedmapTest(imgOutA_P, imgOutB_P, imgInA_P, imgInB_P, imgInA.cols, imgInA.rows, LAPRAD, GAURAD, GAUSIG, GFRAD, GFEPS);

    chrono::steady_clock::time_point stopPoint = chrono::steady_clock::now();
    chrono::duration<double> elapsedTime = chrono::duration_cast<chrono::duration<double>>(stopPoint - startPoint);
    cout << "Test Time : " << elapsedTime.count() * 1000.0 << " ms." << endl;

    imgOutA.convertTo(imgOutA, CV_8UC1, 255.0);
    imshow("Output Image H", imgOutA);
#ifdef TWOIMG
    imgOutB.convertTo(imgOutB, CV_8UC1, 255.0);
    imshow("Output Image L", imgOutB);
#endif

    waitKey(0);

    return 0;
}
