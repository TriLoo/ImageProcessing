//
// Created by smher on 18-12-9.
//

#ifndef RDLW_SAL_FUSIONCUDA_GUIDEDFILTER_H
#define RDLW_SAL_FUSIONCUDA_GUIDEDFILTER_H

#include "headers.h"

namespace IVFusion
{
    class GuidedFilter
    {
    public:
        //GuidedFilter(int rad, double eps) : rad_(rad), eps_(eps){}
        GuidedFilter(int r, int c);
        ~GuidedFilter();

        void doGuidedFilter(float *d_imgOut, const float *d_imgInI, float *d_imgInP, cudaStream_t cs, int rad, double eps);
        void doGuidedFilter(float *d_detail, float *d_base, const cv::Mat& imgInI, const cv::Mat& imgInP, int rad1, double eps1, int rad2, double eps2);
    private:
        float *d_tempA_, *d_tempB_;

        dim3 blockPerGrid, threadPerBlock;

        int rows_, cols_;


        //int rad_;
        //double eps_;

        void bindTexture(float *d_imgInI, float *d_imgInP);
        void releaseTexture();

    };
}     // namespace IVFusion

#endif //RDLW_SAL_FUSIONCUDA_GUIDEDFILTER_H
