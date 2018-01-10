//
// Created by smher on 18-1-10.
//
#include "LocalSaliency.h"

using namespace std;
using namespace cv;

LocalSaliency::LocalSaliency(int r, int c, int ar, int gr, double gs) : row(r), col(c), AvgRad(ar), GauRad(gr), GauSig(gs)
{
}

void LocalSaliency::localSaliency(cv::Mat &sal, const cv::Mat &imgIn)
{
    // prepare temp Mat
    Mat AvgMat(Size(col, row), imgIn.type());
    Mat GauMat(Size(col, row), imgIn.type());

    boxFilter(imgIn, AvgMat, imgIn.depth(), Size(AvgRad, AvgRad));
    GaussianBlur(imgIn, GauMat, Size(GauRad, GauRad), GauSig);

    AvgMat = AvgMat - GauMat;
    GauMat = abs(AvgMat);

    // begin morphological filtering
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));    // 椭圆形状
    morphologyEx(GauMat, AvgMat, MORPH_CLOSE, kernel);

    sal = AvgMat;
}

