#include <iostream>
#include "vector"
#include "string"
#include "cassert"
#include "stdexcept"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat img = imread("Barbara.jpg", IMREAD_COLOR);
    imshow("Input", img);

    waitKey(0);

    return 0;
}
