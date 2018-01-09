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
    RDLWavelet(imgOuts, imgIn);

    // for test
    Mat tempMat;
    imgOuts[1].convertTo(tempMat, CV_8UC1, 255);
    //normalize(imgOuts[0], tempMat, 0.0, 255.0, NORM_MINMAX);
    //tempMat.convertTo(tempMat, CV_8UC1, 1.0);
    imwrite("Output.jpg", tempMat);
    imshow("Output", tempMat);
    //cout << tempMat.cols << endl;
    cout << "Size = " << imgOuts.size() << endl;
    cout << "Output = " << tempMat.rows << " * " ; cout <<  tempMat.cols << endl;

    waitKey(0);

    return 0;
}
