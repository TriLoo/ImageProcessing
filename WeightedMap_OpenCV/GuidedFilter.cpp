//
// Created by smher on 18-1-11.
//

#include "GuidedFilter.h"

GuidedFilter::GuidedFilter(int r, double e) : rad_(r), eps_(e)
{
}

GuidedFilter::~GuidedFilter()
{
}

void GuidedFilter::guidedfilter(cv::Mat &imgOut, const cv::Mat &imgInI, cv::Mat &imgInP)
{
}

