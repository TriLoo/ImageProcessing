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
        GuidedFilter(int rad, double eps) : rad_(rad), eps_(eps){}
        ~GuidedFilter(){}

        void doGuidedFilter(float *d_imgOut, float *d_imgInI, float *d_imgInP);

    private:
        int rad_;
        double eps_;
    };
}     // namespace IVFusion

#endif //RDLW_SAL_FUSIONCUDA_GUIDEDFILTER_H
