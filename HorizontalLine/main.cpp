#include <iostream>
#include "vector"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;
    Mat img = imread("e57KK.jpg", IMREAD_COLOR);
    cvtColor(img, img, CV_BGR2GRAY);
    imshow("Input", img);

    Mat kernel = Mat::ones(Size(100, 1), CV_8UC1);

    Mat res;
    morphologyEx(img, res, MORPH_CLOSE, kernel);

    imshow("Output", res);

    waitKey(0);

    return 0;
}