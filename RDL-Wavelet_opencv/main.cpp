#include "headers.h"
#include "RDL_Wavelet.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat imgIn = imread("Barbara.jpg", IMREAD_COLOR);
    imshow("Input", imgIn);

    if(imgIn.channels() != 1)
        cvtColor(imgIn, imgIn, CV_BGR2GRAY);

    // convert the input image to float
    imgIn.convertTo(imgIn, CV_32F, 1.0/255);

    // prepare output vector
    vector<Mat> imgOuts(0);

    RDLWavelet(imgOuts, imgIn);

    // for test
    Mat tempMat;
    imgOuts[0].convertTo(tempMat, CV_8UC1, 255);
    imshow("Output", tempMat);
    cout << "Output = " << tempMat.rows << " * " << tempMat.cols << endl;

    waitKey(0);

    return 0;
}
