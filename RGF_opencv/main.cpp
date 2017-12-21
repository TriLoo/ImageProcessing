//#include <iostream>
#include "RGF.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    //JBF<float> jf(2, 1.2, 0.25, 1);
    RGF<float> rf(2, 1.2, 0.25, 2);

    //Mat imgIn = Mat::ones(5, 5, CV_32FC3);
    //Mat imgIn = Mat::ones(5, 5, CV_32F);
    //Mat imgIn = Mat_<Vec3f>(Size(5, 5), Vec3f(1, 1, 1));
    //Mat imgOut = Mat::ones(5, 5, CV_32F);
    //Mat imgOut = Mat::ones(5, 5, CV_32FC3);
    vector<Mat> imgOutVec(0);

    Mat imgIn = imread("Barbara.jpg", IMREAD_COLOR);
    Mat imgOut = Mat::zeros(imgIn.size(), imgIn.depth());

    imshow("Input", imgIn);

    try
    {
        if(imgIn.empty())
            throw runtime_error("Read image failed ...");
    }
    catch(runtime_error err)
    {
        cerr << err.what() << endl;
        return -1;
    }


    imgIn.convertTo(imgIn, CV_32FC3, 1.0/255);

    //cout << "imgIn.channel = " << imgIn.channels() << endl;
    //cout << "imgIn = " << imgIn << endl;

    //jf.jointBilateralFilter(imgOut, imgIn, imgIn);
    //rf.jointBilateralFilter(imgOut, imgIn, imgIn);
    rf.rollingguidancefilter(imgOutVec, imgIn);

    cout << "Length of result = " << imgOutVec.size() << endl;

    //cout << imgOutVec[0] << endl;
    imgOutVec[0].convertTo(imgOut, CV_8UC3, 255);

    // show the result
    imshow("RGF Result", imgOut);
    waitKey(0);

    /*    // All work
    Mat_<float> test(3, 3);
    test << 1, 2, 3, 4, 5, 6, 6, 7, 8;
    cout << test << endl;
    Mat test = (Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    cout << test << endl;
    Mat_<float> test(3, 3);
    test << 1, 2, 3, 4, 5, 6, 6, 7, 8;
    for(auto beg = test.begin(); beg != test.end(); beg++)
        cout << *beg << endl;
    for (int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            cout << test(i, j) << endl;

    Mat Ins = Mat::ones(Size(3, 3), CV_32FC1);
    Mat Outs = Mat::ones(Size(3, 3), CV_32FC1);

    cv::exp(Ins, Outs);

    cout << Outs << endl;
    */

    return 0;
}