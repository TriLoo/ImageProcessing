//
// Created by smher on 18-1-11.
//

#include "WeightedMap.h"

using namespace std;
using namespace cv;

WeightedMap::WeightedMap(int r, int c) : row_(r), col_(c)
{
}

WeightedMap::~WeightedMap()
{
}

// calculate the histogram of single channel image
void WeightedMap::hcsingle(cv::Mat &imgOut, const cv::Mat &imgIn)
{

}

// calculation of global saliency map
void WeightedMap::globalsaliency(cv::Mat &imgOut, const cv::Mat &imgIn)
{
    Mat imgTemp(imgIn);
    //assert(imgIn.channels() == CV_32FC1);
    if (imgIn.depth() == CV_32F)
        imgTemp.convertTo(imgTemp, CV_8UC1, 255);

    int dims = 256;
    int histSize[] = {256};
    float ranges[] = {0, 256};
    const float *histRange[] = {ranges};

    //float Kfactor = imgIn.rows * imgIn.cols;
    float Kfactor = row_ * col_;

    bool uniform = true, accumulate = false;

    //cv::Mat hist = cv::Mat_<int>(dims, 1, 0);
    cv::Mat hist;
    cv::calcHist(&imgIn, 1, 0, cv::Mat(), hist, 1, histSize, histRange, uniform, accumulate);

    hist /= Kfactor;

    //std::cout << hist << std::endl;
    //std::vector<float> dist(0);
    cv::Mat lut = cv::Mat_<float>(cv::Size(1, 256), 0);

    for(int i = 0; i < dims; ++i)
    {
        float tempSum = 0.0;
        for (int j = 0; j < dims; ++j)
            tempSum += hist.at<float>(j) * fabs(j - i);
        //dist.push_back(tempSum);
        lut.at<float>(i) = tempSum;
    }

    cv::LUT(imgIn, lut, imgOut);
    cv::normalize(imgOut, imgOut, 0, 1, CV_MINMAX);
}

// calculation of local saliency map
void WeightedMap::localsaliency(cv::Mat &sal, const cv::Mat &imgIn)
{
    //const int col =
    // prepare temp Mat
    Mat AvgMat(Size(col_, row_), imgIn.type());
    Mat GauMat(Size(col_, row_), imgIn.type());

    boxFilter(imgIn, AvgMat, imgIn.depth(), Size(AvgRad_, AvgRad_));
    GaussianBlur(imgIn, GauMat, Size(GauRad_, GauRad_), GauSig_);

    AvgMat = AvgMat - GauMat;
    GauMat = abs(AvgMat);

    // begin morphological filtering
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));    // 椭圆形状
    morphologyEx(GauMat, AvgMat, MORPH_CLOSE, kernel);

    sal = AvgMat;
}

// calculation of guided filter
void WeightedMap::guidedfilter(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
}

// calculation of weighted map
void WeightedMap::weightedmap(cv::Mat &wmBase, cv::Mat &wmDetail, std::vector<cv::Mat> &imgIns)
{

}

