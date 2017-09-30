#include "BFilter.h"
#include "GFilter.h"
#include "guidedfilter.h"

//#define FILTERR 10
//#define EPS 0.01
#define FILTERR 20
#define EPS 0.1

int main() {
    //std::cout << "Hello, World!" << std::endl;

    clock_t start, stop;

    //Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
    Mat img = imread("barbara.jpg", IMREAD_GRAYSCALE);
    try
    {
        if(!img.data)
            throw runtime_error("Read Image failed...");
    }
    catch(runtime_error err)
    {
        cerr << err.what() << endl;
        return -1;
    }

    int row = img.rows;
    int col = img.cols;

    cout << "Image Info: " << row << " * " << col << endl;

    imshow("Input", img);

    img.convertTo(img, CV_32F, 1.0);

    float *imgInP = (float *)img.data;

    Mat imgOut = Mat::zeros(Size(row, col), CV_32F);
    float *imgOutP = (float *)imgOut.data;

    //BFilter bf(col, row);
    //bf.boxfilterTest(imgOutP, imgInP, col, row, FILTERR);
    GFilter gf(col, row);

    start = clock();
    gf.guidedfilterTest(imgOutP, imgInP, imgInP, col, row, FILTERR, EPS);
    stop = clock();
    cout << "GPU Time : " << (stop - start) * 1.0 / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;

    // Output For test
    /*
    for(int i = 0; i < 10; ++i)
        cout << imgOutP[i] << endl;
    */

    imgOut.convertTo(imgOut, CV_8UC1, 1.0);
    imshow("GPU Output", imgOut);

    // guided filter on OpenCV
    Mat OpenCV_ImgOut = Mat::zeros(Size(row, col), CV_32F);
    start = clock();
    OpenCV_ImgOut = guidedFilter(img, img, FILTERR, EPS, CV_32F);
    stop = clock();
    cout << "OpenCV Time : " << (stop - start) * 1.0 / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
    OpenCV_ImgOut.convertTo(OpenCV_ImgOut, CV_8UC1, 1.0);
    imshow("OpenCV Output", OpenCV_ImgOut);

    waitKey(0);

    return 0;
}
