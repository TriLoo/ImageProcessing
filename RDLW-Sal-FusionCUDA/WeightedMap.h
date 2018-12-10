//
// Created by smher on 18-12-9.
//

#ifndef RDLW_SAL_FUSIONCUDA_WEIGHTEDMAP_H
#define RDLW_SAL_FUSIONCUDA_WEIGHTEDMAP_H

#include "headers.h"
#include "GuidedFilter.h"

namespace IVFusion
{
    class GuidedFilter;
    class WeightedMap
    {
    public:
        WeightedMap();
        ~WeightedMap();

        void setParams(double c = 0.95,
                        int ar = 20,
                        int gr = 3,
                       int guiR = 30,
                       double gs = 0.3,
                       double ge = 0.0001
        );
        void doWeightedMap(float *d_wmDetailA, float *d_wmDetailB, float *d_wmBaseA, float *d_wmBaseB, const cv::Mat& imgInA, const cv::Mat& imgInB);
    private:
        cv::Mat salMatA_, salMatB_;
        GuidedFilter *mpGuidedFilter_;

        void localsaliency(cv::Mat& sal, const cv::Mat& imgIn);
        void globalsaliency(cv::Mat& sal, const cv::Mat& imgIn);
        void doSaliencyDetection(cv::Mat& imgSal, cv::Mat& imgIn);
    };

}   // IVFusion

#endif //RDLW_SAL_FUSIONCUDA_WEIGHTEDMAP_H
