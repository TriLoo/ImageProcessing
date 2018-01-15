//
// Created by smher on 18-1-11.
//

#include "WeightedMap.h"

using namespace std;
using namespace cv;

void imgShow(Mat& img)
{
    imshow("Eva", img);
    waitKey(0);
}
void imgShow(Mat&& img)
{
    imshow("Eva", img);
    waitKey(0);
}

WeightedMap::WeightedMap(int r, int c) : row_(r), col_(c)
{
}

WeightedMap::~WeightedMap()
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
    cv::calcHist(&imgTemp, 1, 0, cv::Mat(), hist, 1, histSize, histRange, uniform, accumulate);

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

    cv::LUT(imgTemp, lut, imgOut);
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

// calculation of final saliency map
void WeightedMap::saliencydetection(cv::Mat &sal, const cv::Mat &imgIn)
{
    Mat localSal, globalSal;
    localsaliency(localSal, imgIn);
    globalsaliency(globalSal, imgIn);

    sal = c_ * localSal + (1 - c_) * globalSal;
}

// calculation of guided filter, single channel
void WeightedMap::guidedfilter(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP, int rad, double eps)
{
    assert(imgInI.channels() == 1);      // grayscale input

    Mat meanI, meanP, corrI, corrIp;

    // Step 1 in Algorithm 1
    boxFilter(imgInI, meanI, imgInI.depth(), Size(rad, rad));
    boxFilter(imgInP, meanP, imgInP.depth(), Size(rad, rad));
    boxFilter(imgInI.mul(imgInI), corrI, imgInI.depth(), Size(rad, rad));
    boxFilter(imgInI.mul(imgInP), corrIp, imgInI.depth(), Size(rad, rad));
    //assert(imgInI.type() == imgInP.type());
    //cout << imgInI.depth() << endl;
    //cout << imgInP.depth() << endl;
    //cout << "Success 3.0." << endl;

    // Step 2 in Algorithm 1
    Mat varI = corrI - meanI.mul(meanI);
    Mat covIp = corrIp - meanI.mul(meanP);

    //cout << "Success 3.1." << endl;

    // Step 3 in Algorithm 1
    Mat a = covIp / (varI + eps);
    Mat b = meanP - a.mul(meanI);

    //cout << "Success 3.2." << endl;

    // Step 4
    boxFilter(a, meanI, a.depth(), Size(rad, rad));    // meanI --> meanA
    boxFilter(b, meanP, b.depth(), Size(rad, rad));    // meanP --> meanB

    // Step 5
    imgOut = meanI.mul(imgInI) + meanP;
}

// calculation of weighted map
void WeightedMap::weightedmap(std::vector<cv::Mat> &wmBase, std::vector<cv::Mat> &wmDetail, std::vector<cv::Mat> &imgIns)
{
    wmBase.clear();
    wmDetail.clear();

    //cout << "Success 1." << endl;

    Mat salA, salB;
    saliencydetection(salA, imgIns[0]);
    saliencydetection(salB, imgIns[1]);

    //cout << "Success 2." << endl;

    Mat salMapA, salMapB;
    salMapA = salA >= salB;
    salMapB = salB > salA;

    //normalize(salMapA, salMapA, 0, 1, NORM_MINMAX);
    //normalize(salMapB, salMapB, 0, 1, NORM_MINMAX);
    salMapA.convertTo(salMapA, CV_32F, 1.0/255);
    salMapB.convertTo(salMapB, CV_32F, 1.0/255);

    //cout << "Success 3." << endl;

    int r1 = 30, r2 = 7;
    double eps1 = 10^(-4), eps2 = 10^(-6);

    Mat wmA, wmB, tempMat;
    // base layer
    guidedfilter(wmA, imgIns[0], salMapA, r1, eps1);
    guidedfilter(wmB, imgIns[1], salMapB, r1, eps1);
    tempMat = wmA + wmB;
    wmBase.push_back(wmA / tempMat);
    wmBase.push_back(wmB / tempMat);

    //imgShow(wmA / tempMat);
    //imgShow(wmB / tempMat);
    //cout << "Success 4." << endl;

    // detail layers
    guidedfilter(wmA, imgIns[0], salMapA, r2, eps2);
    guidedfilter(wmB, imgIns[1], salMapB, r2, eps2);
    tempMat = wmA + wmB;
    wmDetail.push_back(wmA / tempMat);
    wmDetail.push_back(wmB / tempMat);

    //imgShow(wmA / tempMat);
    //imgShow(wmB / tempMat);
}

