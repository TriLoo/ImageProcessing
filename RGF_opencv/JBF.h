//
// Created by smher on 17-12-18.
//

#ifndef RGF_SALIENCY_JBF_H
#define RGF_SALIENCY_JBF_H

// 使用模板时，类的声明与定义最好包含在相同的头文件中，否则:undefined reference
#include "header.h"

template <typename T>
class JBF
{
public:
    JBF() = default;
    JBF(int rad, double deltaS, double deltaR, int level);
    virtual ~JBF();

    void jointBilateralFilter(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP);
    void jointBilateralFilter(cv::Mat &imgOut, cv::Mat &imgIn, int rad, double deltaS, double deltaR, int level);
private:
    void JBFSingleChannel(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP);
    void JBFColor(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP);

    // mg: meshgrid, the output
    // rg: the range (element) of output(mg)
    void meshgrid(cv::Range rg, cv::Mat &mg);

    T calSumMat(cv::Mat &Ins);

protected:
    int rad_, level_;
    double deltaS_, deltaR_;
};

template <typename T>
inline T JBF<T>::calSumMat(cv::Mat &Ins)
{
    T sumT = 0;
//#pragma unloop
    for(auto beg = Ins.begin<T>(); beg != Ins.end<T>(); beg++)
        sumT += *beg;
    return sumT;
}

template <typename T>
JBF<T> ::JBF(int rad, double deltaS, double deltaR, int level) : rad_(rad), deltaS_(deltaS), deltaR_(deltaR), level_(level)
{
}

template <typename T>
JBF<T>::~JBF()
{
}

template <typename T>
void JBF<T>::jointBilateralFilter(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    if(imgInI.channels() == 1)
        JBFSingleChannel(imgOut, imgInI, imgInP);
    else
        JBFColor(imgOut, imgInI, imgInP);
}

template <typename T>
void JBF<T>::meshgrid(cv::Range rg, cv::Mat &mg)
{
    std::vector<int> x(0);
    for(int i = rg.start; i <= rg.end; ++i)
        x.push_back(i);
    cv::repeat(cv::Mat(x).t(), x.size(), 1, mg);
}

template <typename T>      // T can be uint or float
void JBF<T>::JBFSingleChannel(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    double multS = 1.0 / (2 * deltaS_ * deltaS_);
    double multR = 1.0 / (2 * deltaR_ * deltaR_);

    const int len = 2 * rad_ + 1;
    cv::Mat gs = cv::Mat_<float>(len, len);
    for(int i = -rad_; i <= rad_; i++)
        for(int j = -rad_; j <= rad_; j++)
            gs.at<float>(i + rad_, j + rad_) = exp((i * i + j * j) * multS);
    // for test
    // std::cout << distPatch << std::endl;

    int row = imgInI.rows, col = imgInI.cols;

    cv::Mat imgInI_T, imgInP_T;
    cv::copyMakeBorder(imgInI, imgInI_T, rad_, rad_, rad_, rad_, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(imgInP, imgInP_T, rad_, rad_, rad_, rad_, cv::BORDER_REPLICATE);

    cv::Mat patch1 = cv::Mat::zeros(cv::Size(len, len), CV_32F);
    cv::Mat patch2 = cv::Mat::zeros(cv::Size(len, len), CV_32F);

    cv::Mat gr = cv::Mat::zeros(cv::Size(len, len), CV_32FC1);
    cv::Mat d = cv::Mat::zeros(cv::Size(len, len), CV_32F);
    //cv::Mat sumCal = cv::Mat::ones(cv::Size(len, len), CV_32F);
    float tempSum = 0.0;
    for (int i = rad_; i < rad_+row; ++i)
    {
        for(int j = rad_; j < rad_ + col; ++j)
        {
            patch1 = imgInP_T(cv::Range(i - rad_, i + rad_ + 1), cv::Range(j - rad_, j + rad_ + 1));
            patch2 = imgInI_T(cv::Range(i - rad_, i + rad_ + 1), cv::Range(j - rad_, j + rad_ + 1));

            //tempVal = imgInI_T.at<float>(i, j);
            //d = cv::Mat(cv::Size(len, len), CV_32F, tempVal);    // 或者下面这种做法
            d = cv::Mat_<float>(cv::Size(len, len), imgInI_T.at<float>(i, j)) - patch2;
            cv::exp(-d * multR, gr);

            gr.mul(gs);

            //tempSum = 1.0 / calSumMat(gr);
            tempSum = 1.0 / cv::sum(gr)[0];

            gr.mul(patch1);

            imgOut.at<float>(i - rad_, j - rad_) = cv::sum(gr)[0] * tempSum;
        }
    }
}

template <typename T>
void JBF<T>::JBFColor(cv::Mat &imgOut, const cv::Mat &imgInI, const cv::Mat &imgInP)
{
    const int len = 2 * rad_ + 1;
    double multS = 1.0 / (2 * deltaS_ * deltaS_);
    double multR = 1.0 / (2 * deltaR_ * deltaR_);

    // for debug
    // std::cout << "Step 0 Success." << std::endl;

    // can use cv::repeat to realize 'meshgrid' in matlab
    //
    int row = imgInI.rows, col = imgInI.cols;
    cv::Mat gs = cv::Mat_<cv::Vec3f>(len, len);
    double tempVal = 0.0;
    for(int i = -rad_; i <= rad_; i++)
        for(int j = -rad_; j <= rad_; j++)
        {
            tempVal = exp((i * i + j * j) * multS);
            //gs.at<cv::Vec3f>(i + rad_, j + rad_) = cv::Vec<float, 3>(tempVal, tempVal, tempVal);
            gs.at<cv::Vec3f>(i + rad_, j + rad_) = cv::Vec3f(tempVal, tempVal, tempVal);
        }

    // std::cout << "Step 1 Success." << std::endl;

    cv::Mat imgInI_T, imgInP_T;
    cv::copyMakeBorder(imgInI, imgInI_T, rad_, rad_, rad_, rad_, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(imgInP, imgInP_T, rad_, rad_, rad_, rad_, cv::BORDER_REPLICATE);

    cv::Mat patch1 = cv::Mat::zeros(cv::Size(len, len), CV_32FC3);
    cv::Mat patch2 = cv::Mat::zeros(cv::Size(len, len), CV_32FC3);

    cv::Mat gr = cv::Mat::zeros(cv::Size(len, len), CV_32FC3);
    cv::Mat d = cv::Mat::zeros(cv::Size(len, len), CV_32FC3);
    //cv::Mat sumCal = cv::Mat::ones(cv::Size(len, len), CV_32F);
    cv::Vec3f tempSum(0, 0, 0);
    cv::Scalar cs;

    // std::cout << "Step 2 Success." << std::endl;

    for(int i = rad_; i < rad_ + row; ++i)
        for (int j = rad_; j < rad_ + col; ++j)
        {
            patch1 = imgInP_T(cv::Range(i - rad_, i + rad_ + 1), cv::Range(j - rad_, j + rad_ + 1));
            patch2 = imgInI_T(cv::Range(i - rad_, i + rad_ + 1), cv::Range(j - rad_, j + rad_ + 1));

            // for test
            // std::cout << patch2.channels() << std::endl;
            // std::cout << patch2 << std::endl;
            // std::cout << "Step 3.1 Success." << std::endl;

            d = cv::Mat_<cv::Vec3f>(cv::Size(len, len), imgInI_T.at<cv::Vec3f>(i, j)) - patch2;
            cv::exp(-d * multR, gr);

            // std::cout << gr.channels() << std::endl;
            // std::cout << "Step 3.2 Success." << std::endl;

            gr.mul(gs);

            cs = cv::sum(gr);
            tempSum = cv::Vec3f(1.0 / cs[0], 1.0 / cs[1], 1.0 / cs[2]);

            // std::cout << "Step 3.3 Success." << std::endl;

            cs = cv::sum(gr.mul(patch1));

            // std::cout << "Step 3.4 Success." << std::endl;

            imgOut.at<cv::Vec3f>(i - rad_, j - rad_) = cv::Vec3f(cs[0] * tempSum[0], cs[1] * tempSum[1], cs[2] * tempSum[2]);

            // std::cout << "Step 3.5 Success." << std::endl;
        }

}

#endif //RGF_SALIENCY_JBF_H
