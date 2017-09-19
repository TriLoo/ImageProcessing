#include "gaussfilter.h"

#define FILTERR 5
#define FILTERW (2 * FILTERR + 1)
//#define FILTERSIZE FILTERW * FILTERW
#define FILTERS 5          // the sigma of gaussian filter

int main()
{
    //Mat img = imread("testA.jpg", IMREAD_GRAYSCALE);
    Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);

    if(!img.data)
    {
        cerr << "Read image failed..." << endl;
        return -1;
    }

    int row = img.rows;
    int col = img.cols;

    cout << "Image Information : " << endl;
    cout << row << " * " << col << endl;

    imshow("Input", img);

    img.convertTo(img, CV_32F, 1.0);

    clock_t start, stop;

    start = clock();

    // test gaussian filter on myOpenCV
    Mat imgOpenCV = Mat::zeros(row, col, CV_32F);
    GaussianBlur(img, imgOpenCV, Size(11, 11), 5, 5, BORDER_CONSTANT);

    stop = clock();
    double dur = (double)((stop - start) * 1.0 / CLOCKS_PER_SEC) * 1000.0;
    cout << "OpenCV : " << dur << " ms" << endl;

    imgOpenCV.convertTo(imgOpenCV, CV_8UC1, 1.0);
    imshow("OpenCV Result", imgOpenCV);


    float *imgIn = (float *)img.data;

    Mat imgOut  = Mat::zeros(row, col, CV_32F);
    //float *imgOutP = reinterpret_cast<float *>(imgOut.data);
    float *imgOutP = (float *)(imgOut.data);

    GFilter gf(row, col, FILTERW, FILTERS);

    start = clock();
    //gf.gaussfilterGlo(imgOutP, imgIn, col, row, nullptr, FILTERW);
    //gf.gaussfilterTex(imgOutP, imgIn, col, row, nullptr, FILTERW);
    //gf.gaussfilterSha(imgOutP, imgIn, col, row, nullptr, FILTERW);
    gf.gaussfilterShaSep(imgOutP, imgIn, col, row, nullptr, nullptr, FILTERW);   // 750 Mpixel / sec

    stop = clock();

    dur = (double)((stop - start) * 1.0 / CLOCKS_PER_SEC) * 1000.0;

    cout << "GPU Gaussian Filtering time : " << dur << " ms" << endl;

    imgOut.convertTo(imgOut, CV_8UC1, 1.0);

    /*
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
            cout << imgOutP[i] << ", ";
        cout << endl;
    }
    */

    imshow("GPU Result", imgOut);

    waitKey(0);

    return 0;
}