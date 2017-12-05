#include <iostream>
#include "cassert"
#include "vector"
#include "string"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv/cv.h"
//#include "memory"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    string imgName = "lena.jpg";
    Mat img = imread(imgName, IMREAD_GRAYSCALE);

    if(!img.data)
    {
        cerr << "Read image failed..." << endl;
        return -1;
    }

    //Mat Res = Mat::zeros(img.size(), CV_8UC1);


    //shared_ptr<float> histRange = make_shared<float>(0, 255);
    float ranges[] = {0, 255};
    const float *histRange[] = {ranges};
    int histSize = 255;
    int channel = 0;
    MatND Hist;
    calcHist(&img, 1, &channel, Mat(), Hist, 1, &histSize, histRange, true, false);

    //cout << Hist << endl;
    //cvEqualizeHist(&img, &Res);
    Mat HistImage = Mat(histSize, histSize, CV_8U, cv::Scalar(255));

    for(int i = 0; i < histSize; i++)
        line(HistImage, Point(i, histSize), Point(i, histSize - Hist.at<uchar>(i)), cv::Scalar::all(0));

    Mat Res;
    equalizeHist(img, Res);

    imshow("Input Image", img);
    imshow("Hist Image: Before", HistImage);
    imshow("Processed", Res);

    calcHist(&Res, 1, &channel, Mat(), Hist, 1, &histSize, histRange, true, false);

    for(int i = 0; i < histSize; i++)
        line(HistImage, Point(i, histSize), Point(i, histSize - Hist.at<uchar>(i)), cv::Scalar::all(0));

    imshow("Hist Image: After", HistImage);

    waitKey(0);

    return 0;
}

