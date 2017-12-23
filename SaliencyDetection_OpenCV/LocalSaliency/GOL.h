//
// Created by smher on 17-12-23.
//

#ifndef LOCALSALIENCYGOL_GOL_H
#define LOCALSALIENCYGOL_GOL_H

#include "headers.h"

template <typename T>
class GOL
{
public:
    GOL() = default;
    GOL(int lrad, cv::Size sc, double dsig);
    ~GOL();

    void gaussoflap(cv::Mat &imgOut, const cv::Mat &imgIn);
private:
    int laprad_;
    cv::Size gauSize_;
    double gausig_;
};

template <typename T>
GOL<T>::GOL(int lrad, cv::Size sc, double dsig) : laprad_(lrad), gauSize_(sc), gausig_(dsig)
{
}

template <typename T>
GOL<T>::~GOL()
{
}

template <typename T>
void GOL<T>::gaussoflap(cv::Mat &imgOut, const cv::Mat &imgIn)
{
    // For test
    //std::cout << imgIn.channels() << std::endl;

    cv::Mat tempImg = cv::Mat::zeros(imgIn.size(), imgIn.depth());

    cv::Laplacian(imgIn, tempImg, imgIn.depth(), laprad_);
    tempImg = cv::abs(tempImg);

    // For test
    //std::cout << tempImg.channels() << std::endl;
    //tempImg.convertTo(tempImg, CV_8UC3, 255.0);
    //cv::imshow("Temp", tempImg);

    cv::GaussianBlur(tempImg, imgOut, gauSize_, gausig_);

    //imgOut.convertTo(tempImg, CV_8UC3, 255.0);
    //cv::imshow("Temp", tempImg);

    // For test
    //std::cout << imgOut.channels() << std::endl;

    if (imgOut.channels() == 3)
        cv::cvtColor(imgOut, imgOut, CV_BGR2GRAY);
        //cv::cvtColor(imgOut, imgOut, CV_RGB2GRAY);
    // For test
    //std::cout << imgOut.channels() << std::endl;

    cv::normalize(imgOut, imgOut, 0.0, 1.0, CV_MINMAX);

    //cvWaitKey(0);
}

#endif //LOCALSALIENCYGOL_GOL_H
