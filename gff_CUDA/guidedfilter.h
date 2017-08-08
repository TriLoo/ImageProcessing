#ifndef __GUIDED_FILTER_
#define __GUIDED_FILTER_
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <

class GFilter
{
public:
    GFilter() = default;
    GFilter(int r, double e):rad(r), eps(e){}
    ~GFilter(){}

    void guidedfilter(cv::Mat &imgI, cv::Mat & imgP);
    void setRad(int r){rad = r;}
    void setEps(double e){eps = e;}
    int readRad(){return rad;}
    double readEps(){return eps;}
private:
    void boxfilter(cv::Mat &imgIn);
    int rad;
    double eps;
};
#endif