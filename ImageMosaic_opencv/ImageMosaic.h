//
// Created by smher on 17-11-4.
//

#ifndef IMAGEMOSAIC_IMAGEMOSAIC_H
#define IMAGEMOSAIC_IMAGEMOSAIC_H

#include "iostream"
#include "cassert"
#include "stdexcept"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
//#include "opencv2/legacy/legacy.hpp"
//#include "opencv/cv.hpp"

using namespace std;
using namespace cv;

typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
}four_corners_t;

class ImageMosaic
{
public:
    ImageMosaic() = default;
    //ImageMosaic(const Mat &H, const Mat &src);
    ~ImageMosaic();


    void FeatureDetectorTest(Mat *out, Mat *in);
    void FeatureMatchTest(vector<Mat *> &out, vector<Mat *> &in);
    void imageMosaicTest(Mat *outs, vector<Mat *> ins);
    void imageRegisterTest(Mat *outs, vector<Mat *> &ins);
private:
    void FeatureDetector(Mat *out, Mat *in);
    void FeatureMatch(vector<Mat *> out, vector<Mat *> in);
    void imageMosaic(Mat *outs, vector<Mat *> &ins);
    void imageRegister(Mat *outs, vector<Mat *> &ins);

    void OptimizeSeam(Mat &img1, Mat &trans, Mat &dst);

    four_corners_t CalcCorners(const Mat &H, const Mat &src);

    four_corners_t corners_;
};

#endif //IMAGEMOSAIC_IMAGEMOSAIC_H
