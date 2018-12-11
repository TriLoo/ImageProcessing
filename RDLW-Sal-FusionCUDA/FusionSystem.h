//
// Created by smher on 18-12-9.
//

#ifndef RDLW_SAL_FUSIONCUDA_FUSIONSYSTEM_H
#define RDLW_SAL_FUSIONCUDA_FUSIONSYSTEM_H

#include "headers.h"
#include "RDLWavelet.h"
#include "WeightedMap.h"

namespace IVFusion
{
    class RDLWavelet;
    class WeightedMap;

    class FusionSystem
    {
    public:
        FusionSystem(int r, int c);
        FusionSystem(FusionSystem& fs) = delete;          // copy construct
        FusionSystem operator=(const FusionSystem& rhs) = delete;
        ~FusionSystem();

        void doFusion(cv::Mat& imgOut, cv::Mat& imgInA, cv::Mat& imgInB);
        void setGFParams(int rad, double eps);

    private:
        int rows_, cols_;

        // point on CPU
        float *h_imgIn_, *h_imgOut_;

        // memory on GPU
        float *d_imgOut_, *d_imgInA_, *d_imgInB_;
        float *d_tempA_, *d_tempB_;  // *d_tempC_, *d_tempD_, *d_tempE_, *d_tempF_, *d_tempG_, *d_tempH_, *d_tempI_, *d_tempM_, *d_tempN_;
        float *d_cA_A_, *d_cH_A_, *d_cV_A_, *d_cD_A_, *d_cA_B_, *d_cH_B_, *d_cV_B_, *d_cD_B_;
        float *d_wmBaseA_, *d_wmBaseB_, *d_wmDetailA_, *d_wmDetailB_;
        // OR:
        //std::vector<float *> d_decomA_;         // 4个分解系数矩阵
        //std::vector<float *> d_decomB_;         // 另一幅图像的4个分解系数矩阵

        // GPU Events
        cudaEvent_t startEvent_, stopEvent_;

        // GPU stream
        cudaStream_t stream1_;

        // GPU status
        cudaError_t cudaStatus_;

        // pimps
        RDLWavelet* mpRDLWavelet_;
        WeightedMap* mpWeightedMap_;

        // Fusion functions based on weighted map & rdlwavelet decomposition matrices.
        void doMatricesFusion();      // based on above pre-allocated memories !
    };
}   // namespace IVFusion

#endif //RDLW_SAL_FUSIONCUDA_FUSIONSYSTEM_H
