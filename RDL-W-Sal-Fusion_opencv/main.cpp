//#include <iostream>
#include "Fusion.h"

using namespace std;
using namespace cv;

int main() {
    //std::cout << "Hello, World!" << std::endl;
    //Mat imgA = imread("source20_1.tif", IMREAD_GRAYSCALE);
    //Mat imgB = imread("source20_2.tif", IMREAD_GRAYSCALE);
    //Mat imgA = imread("Marne_04_IR.bmp", IMREAD_GRAYSCALE);
    //Mat imgB = imread("Marne_04_Vis.bmp", IMREAD_GRAYSCALE);
    //Mat imgA = imread("Balls_IR.bmp", IMREAD_GRAYSCALE);
    //Mat imgB = imread("Balls_Vis.bmp", IMREAD_GRAYSCALE);
    //Mat imgA = imread("IR_lake_g.bmp", IMREAD_GRAYSCALE);
    //Mat imgB = imread("VIS_lake_r.bmp", IMREAD_GRAYSCALE);
    //Mat imgA = imread("Kaptein_1654_IR.bmp", IMREAD_GRAYSCALE);
    //Mat imgB = imread("Kaptein_1654_Vis.bmp", IMREAD_GRAYSCALE);
    //Mat imgA = imread("IR_meting012-1200_g.bmp", IMREAD_GRAYSCALE);
    //Mat imgB = imread("VIS_meting012-1200_r.bmp", IMREAD_GRAYSCALE);
    Mat imgA = imread("TankLWIR.tif", IMREAD_GRAYSCALE);
    Mat imgB = imread("TankVis.tif", IMREAD_GRAYSCALE);

    assert(imgA.empty() != true);
    assert(imgB.empty() != true);

    imshow("Input A", imgA);
    imshow("Input B", imgB);

    imgA.convertTo(imgA, CV_32F, 1.0 / 255);
    imgB.convertTo(imgB, CV_32F, 1.0 / 255);

    const int row = imgA.rows;
    const int col = imgA.cols;
    cout << "Input Image: " << row << " * " << col << endl;

    Mat imgRes;
    Fusion fu(row, col);

    fu.imageFusion(imgRes, imgA, imgB);
    auto start = chrono::steady_clock::now();
    fu.imageFusion(imgRes, imgA, imgB);
    auto stop = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(stop - start);
    cout << "Time used: " << time_used.count() * 1000.0 << " ms." << endl;

    //cout << "Success 3." << endl;

    imshow("Result", imgRes);

    waitKey(0);

    return 0;
}
