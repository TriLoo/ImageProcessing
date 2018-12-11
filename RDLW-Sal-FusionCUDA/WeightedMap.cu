/**
 * @author smh
 * @date 2018.12.10
 *
 * @brief Calculation of weighted map
 *   狂风不终朝，骤雨不终日。
 */

#include "WeightedMap.h"

namespace IVFusion
{
    WeightedMap::WeightedMap(int r, int c): rows_(r), cols_(c)
    {
        threadPerBlock = dim3(BLOCKSIZE, BLOCKSIZE);
        blockPerGrid = dim3(iDiv(c, BLOCKSIZE), iDiv(r, BLOCKSIZE));

        cudaCheckError(cudaMalloc(&d_salA_, sizeof(float) * rows_ * cols_));
        cudaCheckError(cudaMalloc(&d_salB_, sizeof(float) * rows_ * cols_));

        cudaCheckError(cudaStreamCreate(&stream1_));
        cudaCheckError(cudaStreamCreate(&stream2_));

        mpGuidedFilter_ = new GuidedFilter(r, c);
    }

    WeightedMap::~WeightedMap()
    {
        if(d_salA_ != nullptr)
        {
            cudaFree(d_salA_);
            d_salA_ = nullptr;
        }
        if(d_salB_ != nullptr)
        {
            cudaFree(d_salB_);
            d_salB_ = nullptr;
        }

        cudaStreamDestroy(stream1_);
        cudaStreamDestroy(stream2_);
    }

    void WeightedMap::setParams(double c, int ar, int gr, int guiR, double gs, double ge)
    {
        c_ = c;
        AvgRad_ = ar;
        GauRad_ = gr;
        GuiRad_ = guiR;
        GuiEps_ = ge;
    }

    void WeightedMap::globalsaliency(cv::Mat &imgOut, const cv::Mat &imgIn)
    {
        cv::Mat imgTemp(imgIn);
        //assert(imgIn.channels() == CV_32FC1);
        if (imgIn.depth() == CV_32F)
            imgTemp.convertTo(imgTemp, CV_8UC1, 255);

        int dims = 256;
        int histSize[] = {256};
        float ranges[] = {0, 256};
        const float *histRange[] = {ranges};

        //float Kfactor = imgIn.rows * imgIn.cols;
        float Kfactor = rows_ * cols_;

        bool uniform = true, accumulate = false;

        //cv::Mat hist = cv::Mat_<int>(dims, 1, 0);
        cv::Mat hist;
        cv::calcHist(&imgTemp, 1, 0, cv::Mat(), hist, 1, histSize, histRange, uniform, accumulate);

        hist /= Kfactor;

        //std::cout << hist << std::endl;
        //std::vector<float> dist(0);
        cv::Mat lut = cv::Mat_<float>(cv::Size(1, 256), 0);

        for(int i = 0; i < dims; ++i)
        {
            float tempSum = 0.0;
            for (int j = 0; j < dims; ++j)
                tempSum += hist.at<float>(j) * fabs(j - i);
            //dist.push_back(tempSum);
            lut.at<float>(i) = tempSum;
        }

        cv::LUT(imgTemp, lut, imgOut);
        cv::normalize(imgOut, imgOut, 0, 1.0, CV_MINMAX);
    }

    // calculation of local saliency map
    void WeightedMap::localsaliency(cv::Mat &sal, const cv::Mat &imgIn)
    {
        //const int col =
        // prepare temp Mat
        cv::Mat AvgMat(cv::Size(cols_, rows_), imgIn.type());
        cv::Mat GauMat(cv::Size(cols_, rows_), imgIn.type());

        boxFilter(imgIn, AvgMat, imgIn.depth(), cv::Size(AvgRad_, AvgRad_), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
        GaussianBlur(imgIn, GauMat, cv::Size(GauRad_, GauRad_), GauSig_);

        AvgMat = AvgMat - GauMat;
        GauMat = abs(AvgMat);

        // begin morphological filtering
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));    // 椭圆形状
        //Mat kernel = getStructuringElement(MORPH_CROSS, Size(5, 5));    // 椭圆形状
        morphologyEx(GauMat, AvgMat, cv::MORPH_CLOSE, kernel);

        normalize(AvgMat, AvgMat, 0.0, 1.0, cv::NORM_MINMAX);
        sal = AvgMat;
    }

    void WeightedMap::doSaliencyDetection(cv::Mat &imgSal, const cv::Mat &imgIn)
    {
        cv::Mat localSal, globalSal;
        localsaliency(localSal, imgIn);
        globalsaliency(globalSal, imgIn);

        imgSal = c_ * localSal + (1 - c_) * globalSal;
    }

    __global__ void CudaNormalizeWM(float *d_salA_, float *d_salB_, int rows, int cols)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;

        if(idx >= cols || idy >= rows)
            return ;

        float sumAB = d_salA_[INDX(idy, idx, cols)] + d_salB_[INDX(idy, idx, cols)];
        d_salA_[INDX(idy, idx, cols)] /= sumAB;
        d_salB_[INDX(idy, idx, cols)] /= sumAB;
    }

    void WeightedMap::doWeightedMap(float *d_wmDetailA, float *d_wmDetailB, float *d_wmBaseA, float *d_wmBaseB, const float *d_imgInA, const float *d_imgInB,
                                    const cv::Mat &imgInA, const cv::Mat &imgInB)
    {
        // Step 1: do saliency detection
        doSaliencyDetection(salMatA_, imgInA);
        doSaliencyDetection(salMatB_, imgInB);

        // Step 2: construct initial weighted map
        cv::Mat salMapA = cv::Mat::zeros(imgInA.size(), CV_32FC1);
        cv::Mat salMapB = cv::Mat::zeros(imgInB.size(), CV_32FC1);

        for(int i = 0; i < salMatA_.rows; ++i)
        {
            for(int j = 0; j < salMatA_.cols; ++j)
            {
                if(salMatA_.at<float>(i, j) > salMatA_.at<float>(i, j))
                    salMapA.at<float>(i, j) = 1.0;
                else
                    salMapB.at<float>(i, j) = 1.0;
            }
        }

        // Step 3: do guided filter to original weighted map
        int r1 = 30, r2 = 7;
        double eps1 = 1e-4, eps2 = 1e-6;

        /**
        mpGuidedFilter_->doGuidedFilter(d_wmDetailA, d_wmBaseA, imgInA, salMapA, r1, eps1, r2, eps2);
        mpGuidedFilter_->doGuidedFilter(d_wmDetailB, d_wmBaseB, imgInB, salMapB, r1, eps1, r2, eps2);
        */

        // copy image from host to  device
        float *h_salMapA = (float *)salMapA.data;
        float *h_salMapB = (float *)salMapB.data;

        cudaCheckError(cudaHostRegister(h_salMapA, sizeof(float) * rows_ * cols_, cudaHostRegisterDefault));
        cudaCheckError(cudaHostRegister(h_salMapB, sizeof(float) * rows_ * cols_, cudaHostRegisterDefault));

        cudaCheckError(cudaMemcpyAsync(d_salA_, h_salMapA, sizeof(float) * rows_ * cols_, cudaMemcpyHostToDevice, stream1_));
        cudaCheckError(cudaMemcpyAsync(d_salB_, h_salMapB, sizeof(float) * rows_ * cols_, cudaMemcpyHostToDevice, stream2_));


        mpGuidedFilter_->doGuidedFilter(d_wmBaseA, d_imgInA, d_salA_, stream1_, r1, eps1);
        mpGuidedFilter_->doGuidedFilter(d_wmBaseB, d_imgInB, d_salB_, stream2_, r1, eps1);

        mpGuidedFilter_->doGuidedFilter(d_wmDetailA, d_imgInA, d_salA_, stream1_, r2, eps2);
        mpGuidedFilter_->doGuidedFilter(d_wmDetailB, d_imgInB, d_salB_, stream2_, r2, eps2);

        // Step 4: normalize the weighted map
        CudaNormalizeWM<<<blockPerGrid, threadPerBlock, 0, stream1_>>>(d_wmBaseA, d_wmBaseB, rows_, cols_);
        CudaNormalizeWM<<<blockPerGrid, threadPerBlock, 0, stream2_>>>(d_wmDetailA, d_wmDetailB, rows_, cols_);

        // Step 5: clean the host register memory
        cudaCheckError(cudaHostUnregister(h_salMapA));
        cudaCheckError(cudaHostUnregister(h_salMapB));
    }
}      // namespace IVFusion

