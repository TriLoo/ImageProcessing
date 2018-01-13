//
// Created by smher on 18-1-11.
//

#include "WeightedMap.h"

WeightedMap::~WeightedMap()
{
}

// calculate the histogram of single channel image
void WeightedMap::hcsingle(cv::Mat &imgOut, const cv::Mat &imgIn)
{

}

// calculation of global saliency map
void WeightedMap::globalsaliency(cv::Mat &imgOut, cv::Mat &imgIn)
{

}

// calculation of local saliency map
void WeightedMap::localsaliency(cv::Mat &sal, const cv::Mat &imgIn, int AvgRad, int GauRad, double GauSig)
{

}

// calculation of guided filter
void WeightedMap::guidedfilter(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP, int rad, int eps)
{

}

// calculation of weighted map
void WeightedMap::weightedmap(cv::Mat &wmBase, cv::Mat &wmDetail, std::vector<cv::Mat> &imgIns)
{

}

