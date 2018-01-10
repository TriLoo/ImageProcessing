#include "LocalSaliency.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;
    Mat imgIn = imread("Barbara.jpg", IMREAD_GRAYSCALE);

    imshow("Input", imgIn);

    imgIn.convertTo(imgIn, CV_32F, 1.0/255);

    Mat imgSal(Size(imgIn.cols, imgIn.rows), imgIn.type());
    LocalSaliency ls(imgIn.rows, imgIn.cols);
    ls.localSaliency(imgSal, imgIn);

    imgSal.convertTo(imgSal, CV_8UC1, 255.0);
    imshow("Local Saliency", imgSal);


    waitKey(0);

    return 0;
}