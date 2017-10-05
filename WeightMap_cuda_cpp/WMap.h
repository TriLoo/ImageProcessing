//
// Created by smher on 17-10-3.
//

#ifndef WEIGHTEDMAP_WMAP_H
#define WEIGHTEDMAP_WMAP_H

#include "headers.h"

/*
class Filter
{
public:
    Filter() = default;
    Filter(int wid, int hei);
    ~Filter();

    void filter(float *d_imgOut, float *d_imgIn, int wid, int hei, float *d_filter, int rad);
private:

};
*/
//class WMap : public Filter
class WMap
{
public:
    WMap() = default;
    WMap(int wid, int hei, int lr, int gr);
    ~WMap();

    void weightedmap(float *d_imgOut, float *d_imgIn, int wid, int hei, int lr, int gr, int gsigma, int guir, double eps);
    void weightedmapTest(float imgOut, float imgIn, int wid, int hei, int lr, int gr, int gsigma, int guir, double eps);

    void laplacianAbsTest(float *imgOut, float *imgIn, int wid, int hei, int lr);
    void gaussianTest(float *imgOut, float *imgIn, int wid, int hei, int gr, int gsigma);
    void guidedfilterTest(float *imgOut, float *imgIn, int wid, int hei, int guir, double eps);
    void saliencymapTest(float *imgOut, float *imgIn, int wid, int hei, int lr, int gr, double gsigma);

private:
    void laplacianAbs(float *d_imgOut, float *d_imgIn, int wid, int hei, int lr);
    void gaussian(float *d_imgOut, float *d_imgIn, int wid, int hei, int gr, int gsigma);
    void guidedfilter(float *d_imgOut, float *d_imgInI, float *d_imgInP, int wid, int hei, int guir, double eps);
    void saliencymap(float *d_imgOut, float *d_imgIn, int wid, int hei, int lr, int gr, double gsigma);

    int lrad_, grad_;
    float *d_lap_, *d_gau_;
};

#endif //WEIGHTEDMAP_WMAP_H
