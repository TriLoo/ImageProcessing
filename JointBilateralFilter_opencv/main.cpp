//#include <iostream>
#include "JBF.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    JBF<float> jf(2, 1.2, 0.25, 1);

    //Mat imgIn = Mat::ones(5, 5, CV_32FC3);
    //Mat imgIn = Mat::ones(5, 5, CV_32F);
    Mat imgIn = Mat_<Vec3f>(Size(5, 5), Vec3f(1, 1, 1));
    //Mat imgOut = Mat::ones(5, 5, CV_32F);
    Mat imgOut = Mat::ones(5, 5, CV_32FC3);

    //cout << "imgIn.channel = " << imgIn.channels() << endl;
    //cout << "imgIn = " << imgIn << endl;

    jf.jointBilateralFilter(imgOut, imgIn, imgIn);

    cout << imgOut << endl;

    /*
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