/**
 * @author smh
 * @date 2018.12.10
 *
 * @brief Infrared & Visible images fusion
 *   和其光，同其尘。
*/

#include "FusionSystem.h"

void __cudaCheckError(cudaError_t err, const char *filename, const int line)
{
    if(err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << ": in " << filename << ". At " << line << " line." << std::endl;
    }
}

int iDiv(int a, int b)
{
    if(a % b == 0)
        return a / b;
    else
        return a / b + 1;
}

namespace IVFusion
{
    FusionSystem::FusionSystem(int r, int c) : rows_(r), cols_(c), cudaStatus_(cudaSuccess)
    {
        // pre-allocate memory on gpu
        cudaCheckError(cudaMalloc(&d_imgInA_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_imgInB_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_imgOut_, sizeof(float) * r * c));

        cudaCheckError(cudaMalloc(&d_tempA_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempB_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_cA_A_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_cH_A_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_cV_A_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_cD_A_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_cA_B_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_cH_B_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_cV_B_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_cD_B_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_wmBaseA_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_wmBaseB_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_wmDetailA_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_wmDetailB_, sizeof(float) * r * c));
        /**
        cudaCheckError(cudaMalloc(&d_tempA_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempB_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempC_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempD_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempE_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempF_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempH_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempI_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempM_, sizeof(float) * r * c));
        cudaCheckError(cudaMalloc(&d_tempN_, sizeof(float) * r * c));
        */

        // create cuda event
        cudaCheckError(cudaEventCreate(&startEvent_));
        cudaCheckError(cudaEventCreate(&stopEvent_));

        // create cuda stream
        cudaCheckError(cudaStreamCreate(&stream1_));

        // create used other classes
        mpRDLWavelet_ = new RDLWavelet(r, c, 4);
        mpWeightedMap_ = new WeightedMap(r, c);
        mpWeightedMap_->setParams();
    }

    FusionSystem::~FusionSystem()
    {
        // destroy allocated gpu memory
        if(d_imgInA_ != nullptr)
        {
            cudaFree(d_imgInA_);
            d_imgInA_ = nullptr;
        }
        if(d_imgInB_ != nullptr)
        {
            cudaFree(d_imgInB_);
            d_imgInB_ = nullptr;
        }
        if(d_imgOut_ != nullptr)
        {
            cudaFree(d_imgOut_);
            d_imgOut_ = nullptr;
        }

        if(d_tempA_ != nullptr)
        {
            cudaFree(d_tempA_);
            d_tempA_ = nullptr;
        }
        if(d_tempB_ != nullptr)
        {
            cudaFree(d_tempB_);
            d_tempB_ = nullptr;
        }

        if(d_cA_A_ != nullptr)
        {
            cudaFree(d_cA_A_);
            d_cA_A_ = nullptr;
        }
        if(d_cH_A_ != nullptr)
        {
            cudaFree(d_cH_A_);
            d_cH_A_ = nullptr;
        }
        if(d_cV_A_ != nullptr)
        {
            cudaFree(d_cV_A_);
            d_cV_A_ = nullptr;
        }
        if(d_cD_A_ != nullptr)
        {
            cudaFree(d_cD_A_);
            d_cD_A_ = nullptr;
        }
        if(d_cA_B_ != nullptr)
        {
            cudaFree(d_cA_B_);
            d_cA_B_ = nullptr;
        }
        if(d_cH_B_ != nullptr)
        {
            cudaFree(d_cH_B_);
            d_cH_B_ = nullptr;
        }
        if(d_cV_B_ != nullptr)
        {
            cudaFree(d_cV_B_);
            d_cV_B_ = nullptr;
        }
        if(d_cD_B_ != nullptr)
        {
            cudaFree(d_cD_B_);
            d_cD_B_ = nullptr;
        }
        if(d_wmBaseA_ != nullptr)
        {
            cudaFree(d_wmBaseA_);
            d_wmBaseA_ = nullptr;
        }
        if(d_wmBaseB_ != nullptr)
        {
            cudaFree(d_wmBaseB_);
            d_wmBaseB_ = nullptr;
        }
        if(d_wmDetailA_ != nullptr)
        {
            cudaFree(d_wmDetailA_);
            d_wmDetailA_ = nullptr;
        }
        if(d_wmDetailB_ != nullptr)
        {
            cudaFree(d_wmDetailB_);
            d_wmDetailB_ = nullptr;
        }
        /**
        if(d_tempC_ != nullptr)
        {
            cudaFree(d_tempC_);
            d_tempC_ = nullptr;
        }
        if(d_tempD_ != nullptr)
        {
            cudaFree(d_tempD_);
            d_tempD_ = nullptr;
        }
        if(d_tempE_ != nullptr)
        {
            cudaFree(d_tempE_);
            d_tempE_ = nullptr;
        }
        if(d_tempF_ != nullptr)
        {
            cudaFree(d_tempF_);
            d_tempF_ = nullptr;
        }
        if(d_tempG_ != nullptr)
        {
            cudaFree(d_tempG_);
            d_tempG_ = nullptr;
        }
        if(d_tempH_ != nullptr)
        {
            cudaFree(d_tempH_);
            d_tempH_ = nullptr;
        }
        if(d_tempI_ != nullptr)
        {
            cudaFree(d_tempI_);
            d_tempI_ = nullptr;
        }
        if(d_tempM_ != nullptr)
        {
            cudaFree(d_tempM_);
            d_tempM_ = nullptr;
        }
        if(d_tempN_ != nullptr)
        {
            cudaFree(d_tempN_);
            d_tempN_ = nullptr;
        }
        */

        // destroy event, stream etc.
        cudaCheckError(cudaEventDestroy(startEvent_));
        cudaCheckError(cudaEventDestroy(stopEvent_));
        cudaCheckError(cudaStreamDestroy(stream1_));
    }

    void FusionSystem::setGFParams(int rad, double eps)
    {
    }

    //void FusionSystem::doFusion(cv::Mat &imgOut, cv::Mat &imgInA, cv::Mat &imgInB)
    void FusionSystem::doFusion(cv::Mat &imgOut, cv::Mat &imgInA, cv::Mat &imgInB)
    {
        /**
        // test whole fusion system
        float *h_imgInA = (float *)imgInA.data;
        float *h_imgInB = (float *)imgInB.data;
        float *h_imgOut = (float *)imgOut.data;

        std::cout << "image size in fs: " << rows_ << " * " << cols_ << std::endl;
        cudaCheckError(cudaMemcpy(d_imgInA_, h_imgInA, sizeof(float) * rows_ * cols_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_imgInB_, h_imgInB, sizeof(float) * rows_ * cols_, cudaMemcpyHostToDevice));

        // Step 1: do RDLWavelet decomposition to two input images
        mpRDLWavelet_->doRDLWavelet(d_cD_A_, d_cV_A_, d_cH_A_, d_cA_A_, d_imgInA_);
        mpRDLWavelet_->doRDLWavelet(d_cD_B_, d_cV_B_, d_cH_B_, d_cA_B_, d_imgInB_);

        // Step 2: generate weighted map
        mpWeightedMap_->doWeightedMap(d_wmDetailA_, d_wmDetailB_, d_wmBaseA_, d_wmBaseB_, imgInA, imgInB);

        // Step 3: Fusion the matrices
        // TODO


        // Step 4: inverse RDLWavelet
        mpRDLWavelet_->doInverseRDLWavelet(d_imgOut_, d_cD_A_, d_cV_A_, d_cH_A_, d_cA_A_);
        */







        // test Weighted Map
        mpWeightedMap_->doWeightedMap(d_wmDetailA_, d_wmDetailB_, d_wmBaseA_, d_wmBaseB_, d_imgInA_, d_imgInB_, imgInA, imgInB);

        /**
        // test RDLWavelet, AC
        float *h_imgInA = (float *)imgInA.data;
        float *h_imgOut = (float *)imgOut.data;

        std::cout << "image size in fs: " << rows_ << " * " << cols_ << std::endl;
        cudaCheckError(cudaMemcpy(d_imgInA_, h_imgInA, sizeof(float) * rows_ * cols_, cudaMemcpyHostToDevice));

        mpRDLWavelet_->doRDLWavelet(d_cD_A_, d_cV_A_, d_cH_A_, d_cA_A_, d_imgInA_);
        mpRDLWavelet_->doInverseRDLWavelet(d_imgOut_, d_cD_A_, d_cV_A_, d_cH_A_, d_cA_A_);

        //cudaCheckError(cudaMemcpy(h_imgOut, d_cA_A_, sizeof(float) * rows_ * cols_, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(h_imgOut, d_imgOut_, sizeof(float) * rows_ * cols_, cudaMemcpyDeviceToHost));

        cudaDeviceSynchronize();
        */
    }

}   // IVFusion

