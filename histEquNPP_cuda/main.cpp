//#include <iostream>
#include "histEqu.h"

//extern int histEquNpp(Mat &, Mat &);

int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat imgIn = imread("lena.jpg");

    assert(imgIn.data != NULL);

    Mat imgOut = Mat::zeros(imgIn.size(), imgIn.type());     // depth() or type()

    //histEquNpp(imgOut, imgIn);
    histEqu he;
    cout << he.histEquNpp(imgOut, imgIn) << endl;

    return 0;
}