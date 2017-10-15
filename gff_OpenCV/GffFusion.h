//
// Created by smher on 17-10-10.
//

#ifndef GFF_GFFFUSION_H
#define GFF_GFFFUSION_H

#include "iostream"
#include "vector"
#include <ctime>
#include <cassert>
#include <stdexcept>
#include "regex"
#include "boost/format.hpp"
#include <string>

#include <initializer_list>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "guidedfilter.h"

using namespace std;
using namespace cv;

#define KSIZE (Size(31, 31))    // the size of average filter
#define Sigma 5                 // the standard deviation in X direction In Gaussian

class GffFusion
{
public:
    GffFusion() = default;
    void gffFusion(Mat &imgA, Mat &imgB, Mat &Res);
    void gffFusionColor(Mat &imgA, Mat &imgB, Mat &Res);
private:
    void TwoScale(const Mat &inA, Size ksize, Mat &outH, Mat &outL);
    void SaliencyMap(Mat &imgIn, Mat &SMap);
    void WeightMap(Mat &imgInA, Mat &imgInB, vector<Mat *> &vecA, vector<Mat *> &vecB, vector<Mat *> &vec);
    void TwoScaleRec(vector<Mat *> &vecW, vector<Mat *> &LayerIn, Mat &LayerOut);
};

#endif //GFF_GFFFUSION_H
