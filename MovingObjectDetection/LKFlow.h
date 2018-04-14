//
// Created by smher on 18-4-4.
//

#ifndef MOVINGOBJECTDETECTS_LKFLOW_H
#define MOVINGOBJECTDETECTS_LKFLOW_H

#include "headers.h"

class LK
{
public:
    //LK(int l = 3);
    LK();
    ~LK();

    void clacKps(const cv::Mat& imgA, const cv::Mat& imgB);   // calculate keypoints of image A & image B

    // TODO: add pyramid to the calculation
    //void calcPyramid(const cv::Mat &imgA, const cv::Mat &imgB);      // 建立金字塔
    void calcOF(cv::Mat &imgOut, const cv::Mat &imgInA, const cv::Mat& imgInB);
    cv::Point2f calcUV(const cv::Mat& winLeft, const cv::Mat& winRight);      // to calculate the u & v basing on a 3 * 3 window

private:
    //int level_;   // the level of pyramid decomposition
    //std::vector<int> winSize_ = {5, 3, 2, 1};
    std::vector<cv::KeyPoint> kpsA_, kpsB_;  // KeyPoints of image A & image B
    std::vector<cv::Mat> pyramidA_, pyramidB_;    // Image Gaussian Pyramid
    // std::vector<cv::Point2f> glow_(100, cv::Point2f(0, 0));       // Storing the estimation of flow: [g_x, g_y], cannot do this in a class
    // To store the u & v of each keypoint
    std::vector<cv::Point2f> glow_ = std::vector<cv::Point2f>(1, cv::Point2f(0, 0));   // Instead, you should use = in a class ! !

    void calcKps(const cv::Mat& imgA, const cv::Mat& imgB);

    //cv::Mat calcGaussSubsample(const cv::Mat& imgIn, int l);
};

#endif //MOVINGOBJECTDETECTS_LKFLOW_H
