#include "headers.h"
#include "RDL_Wavelet.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat imgIn = imread("Marne_04_IR.bmp", IMREAD_COLOR);
    imshow("Input", imgIn);

    if(imgIn.channels() != 1)
        cvtColor(imgIn, imgIn, CV_BGR2GRAY);

    // convert the input image to float
    imgIn.convertTo(imgIn, CV_32F, 1.0/255);

    /*
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
            cout << imgIn.at<float>(i,j) << ", ";
        cout << endl;
    }
    cout << "--------------------------" << endl;
    */

    // prepare output vector
    vector<Mat> imgOuts(0);

    // begin test
    // Forward Transform
    RDLWavelet rw(imgIn.rows, imgIn.cols);
    rw.RdlWavelet(imgOuts, imgIn);

    // Backward Transform
    Mat imgRes(Size(imgIn.cols, imgIn.rows), CV_32F);
    rw.inverseRdlWavelet(imgRes, imgOuts);

    // for test
    //cout << imgRes(Range(0, 10), Range(0, 10)) << endl;

    // for test
    Mat tempMat;
    imgOuts[0].convertTo(tempMat, CV_8UC1, 255);
    //tempMat.convertTo(tempMat, CV_8UC1, 1.0);
    imwrite("Output.jpg", tempMat);
    imshow("Output", tempMat);
    //cout << tempMat.cols << endl;
    cout << "Size = " << imgOuts.size() << endl;
    cout << "Output = " << tempMat.rows << " * " ; cout <<  tempMat.cols << endl;

    //normalize(imgRes, imgRes, 0, 255, NORM_MINMAX);
    //imgRes.convertTo(imgRes, CV_8UC1, 1.0);
    imgRes.convertTo(imgRes, CV_8UC1, 255.0);
    imshow("Res Image", imgRes);

    waitKey(0);

    return 0;
}

