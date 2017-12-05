//
// Created by smher on 17-12-5.
//
#include <iostream>
#include "cassert"
#include "vector"
#include "string"
#include "math.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

int main()
{
    cout << "Hello world..." << endl;

    Mat img = imread("origin.jpg", IMREAD_COLOR);

    if(!img.data)
    {
        cerr << "Read image failed ..." << endl;
        return -1;
    }

    imshow("Input Image: Processing", img);

    Mat hsv;
    //do transform in HSV space
    cvtColor(img, hsv, COLOR_RGB2HSV);

    int channs = img.channels();
    assert(channs == 3);
    Mat imgRGB[3];

    split(img, imgRGB);

    for(int i = 0; i < 3; i++)
    {
        equalizeHist(imgRGB[i], imgRGB[i]);
    }

    Mat Res;
    merge(imgRGB, channs, Res);
    imshow("Image: Hist Equalize", Res);

    // do laplacian enhancement
    Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

    filter2D(img, Res, img.depth(), kernel, Point(0, 0));

    imshow("Image: Lap Enhance", Res);

    // do log enhancement
    Mat LogImg = Mat::zeros(img.size(), CV_32FC3);
    for(int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
        {
            LogImg.at<Vec3f>(i, j)[0] = log(1 + img.at<Vec3b>(i, j)[0]);
            LogImg.at<Vec3f>(i, j)[1] = log(1 + img.at<Vec3b>(i, j)[1]);
            LogImg.at<Vec3f>(i, j)[2] = log(1 + img.at<Vec3b>(i, j)[2]);
        }

    normalize(LogImg, LogImg, 0, 255, NORM_MINMAX);
    convertScaleAbs(LogImg, LogImg);
    imshow("Image: Log Enhance", LogImg);

    // Gamma Transform
    Mat GamImg = Mat::zeros(img.size(), CV_32FC3);

    for(int i = 0; i < img.rows; i++)
        for(int j = 0; j < img.cols; ++j)
        {
            GamImg.at<Vec3f>(i, j)[0] = pow(img.at<Vec3b>(i, j)[0], 1.5);
            GamImg.at<Vec3f>(i, j)[1] = pow(img.at<Vec3b>(i, j)[1], 1.5);
            GamImg.at<Vec3f>(i, j)[2] = pow(img.at<Vec3b>(i, j)[2], 1.5);
        }

    normalize(GamImg, GamImg, 0, 255, NORM_MINMAX);
    convertScaleAbs(GamImg, GamImg);
    imshow("Image: Gamma Enhance", GamImg);

    // Histogram Specifications
    Mat imgA = imread("lena.jpg", IMREAD_GRAYSCALE);
    Mat imgB = imread("origin2.jpg", IMREAD_GRAYSCALE);

    Mat histA, histB;
    float ranges[] = {0, 255};
    const float *histRange[] = {ranges};
    int histSize = 256;
    channs = 0;
    calcHist(&imgA, 1, &channs, Mat(), histA, 1, &histSize, histRange, true, false);
    calcHist(&imgB, 1, &channs, Mat(), histB, 1, &histSize, histRange, true, false);

    histA /= (imgA.rows * imgA.cols);
    histB /= (imgB.rows * imgB.cols);

    //cout << histA << endl;

    float cdfA[256] = {0};
    float cdfB[256] = {0};

    for(int i = 0; i < histSize; i++)
    {
        if(i == 0)
        {
            cdfA[i] = histA.at<float>(i);
            cdfB[i] = histB.at<float>(i);
        }
        else
        {
            cdfA[i] = cdfA[i - 1] + histA.at<float>(i);
            cdfB[i] = cdfB[i - 1] + histB.at<float>(i);
        }
    }

    /*
    for (int k = 0; k < 10; ++k) {
        cout << "CDFA=" << cdfA[255] << endl;
    }
    */

    //imshow("lena", imgB);

    // calculate the accumulate errors
    float diffError[256][256];
    for(int i = 0; i < histSize; i++)
        for (int j = 0; j < histSize; ++j)
            diffError[i][j] = fabs(cdfA[i] - cdfB[j]);

    // generate LUT
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < histSize; ++i)
    {
        float minVal = diffError[i][0];
        int index = 0;
        for (int j = 0; j < histSize; ++j)
        {
            if (minVal > diffError[i][j])
            {
                minVal = diffError[i][j];
                index = j;
                //continue;
            }
        }
        lut.at<uchar>(i) = static_cast<uchar>(index);
        //if(i < 10)
            //cout << "Index = " << index << endl;
    }

    Mat histSp;
    LUT(imgA, lut, histSp);
    //for (int i = 0; i < 10; ++i) {
        //cout << lut.at<float>(i) << endl;
        //cout << histSp.at<float>(i) << endl;
    //}
    normalize(histSp, histSp, 0, 255, NORM_MINMAX);
    convertScaleAbs(histSp, Res);
    imshow("Image: Hist Sp", histSp);

    waitKey(0);

    return 0;
}
