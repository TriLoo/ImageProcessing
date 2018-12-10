/**
 * @author smh
 * @date 2018.12.10
 *
 * @brief Top frame and algorithm
 *
 * @copyright 2018 smh
 *   license GPL-v3
 */
#include "headers.h"
#include "FusionSystem.h"

using namespace std;
using namespace cv;

int main() {
    //std::cout << "Hello, World!" << std::endl;
    Mat imgInA = imread("Kaptein_1654_IR.bmp", IMREAD_GRAYSCALE);
    Mat imgInB = imread("Kaptein_1654_Vis.bmp", IMREAD_GRAYSCALE);

    Mat imgOut = Mat::zeros(imgInA.size(), CV_32FC1);

    assert(!imgInA.empty());
    assert(!imgInB.empty());
    assert(!imgOut.empty());
    assert(imgInA.isContinuous());
    assert(imgInB.isContinuous());
    assert(imgOut.isContinuous());

    if(imgInA.type() != CV_32FC1)
        imgInA.convertTo(imgInA, CV_32FC1, 1.0/255);
    if(imgInB.type() != CV_32FC1)
        imgInB.convertTo(imgInB, CV_32FC1, 1.0/255);

    int rows = imgInA.rows;
    int cols = imgInA.cols;

    cout << "Image info: " << rows << " * " << cols << endl;

    // test whole fusion framework



    // test for FusionSystem + RDLWavelet
    IVFusion::FusionSystem fs(rows, cols);
    fs.doFusion(imgOut, imgInA, imgInB);

    imshow("test", imgOut);
    waitKey();

    return 0;
}

