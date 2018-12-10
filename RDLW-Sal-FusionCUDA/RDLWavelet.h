//
// Created by smher on 18-12-9.
//

#ifndef RDLW_SAL_FUSIONCUDA_RDLWAVELET_H
#define RDLW_SAL_FUSIONCUDA_RDLWAVELET_H

#include "headers.h"

namespace IVFusion       // infrared - visible image fusion
{
    class RDLWavelet
    {
    public:
        RDLWavelet(int r, int c, int d, cudaStream_t cs = 0);
        ~RDLWavelet();

        void doRDLWavelet(float *d_cD, float *d_cV, float *d_cH, float *d_cA, float *d_imgIn);
        void doInverseRDLWavelet(float *d_imgOut, float *d_cD, float *d_cV, float *d_cH, float *d_cA);
    private:
        int rows_, cols_, dir_;
        dim3 blockPerGrid_, threadPerBlock_;
        cudaStream_t stream1_;

        float *d_Sinc_, *d_temp_;

        void HorizontalPredict(float *d_imgOut, const float * d_imgIn);
        void VerticalPredict(float *d_imgOut, const float *d_imgIn);
        void HorizontalUpdate(float *d_imgOut, const float * d_imgDetail, const float * d_imgIn);
        void VerticalUpdate(float *d_imgOut, const float *d_imgDetail, const float *d_imgIn);
        void InverseHorizontalUpdate(float *d_imgOut, const float * d_imgBase, const float * d_imgDetail);
        void InverseVerticalUpdate(float *d_imgOut, const float * d_imgBase, const float * d_imgDetail);
    };
}   // namespace IVFusion

#endif //RDLW_SAL_FUSIONCUDA_RDLWAVELET_H
