#include "gaussfilter.h"

#define FILTERR 5
#define FILTERW (2 * FILTERR + 1)
//#define FILTERSIZE FILTERW * FILTERW
#define FILTERS 5          // the sigma of gaussian filter

int main()
{
    Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);

    if(!img.data)
    {
        cerr << "Read image failed..." << endl;
        return -1;
    }

    int row = img.rows;
    int col = img.cols;

    imshow("Input", img);

    img.convertTo(img, CV_32F, 1.0);

    float *imgIn = (float *)img.data;

    Mat imgOut  = Mat::zeros(row, col, CV_32F);
    //float *imgOutP = reinterpret_cast<float *>(imgOut.data);
    float *imgOutP = (float *)(imgOut.data);

    GFilter gf(row, col, FILTERW, FILTERS);

    clock_t start, stop;

    start = clock();

    gf.gaussfilterGlo(imgOutP, imgIn, col, row, nullptr, FILTERW);
    //gf.gaussfilterTex(imgOutP, imgIn, col, row, nullptr, FILTERW);
    gf.gaussfilterSha(imgOutP, imgIn, col, row, nullptr, FILTERW);

    stop = clock();

    double dur = (double)((stop - start) / CLOCKS_PER_SEC) * 1000.0;

    cout << "Gaussian Filtering time : " << dur << " ms" << endl;

    imgOut.convertTo(imgOut, CV_8UC1, 1.0);

    /*
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
            cout << imgOutP[i] << ", ";
        cout << endl;
    }
    */

    imshow("Output", imgOut);

    waitKey(0);

    return 0;
}