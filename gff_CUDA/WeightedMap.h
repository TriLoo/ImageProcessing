//
// Created by smher on 17-11-18.
//

#ifndef GFFFUSIONFINAL_WEIGHTEDMAP_H
#define GFFFUSIONFINAL_WEIGHTEDMAP_H

#include "GFilter.h"

class WMap : public GFilter
{
public:
    //using GFilter::GFilter;
    WMap() = default;
    WMap(int wid, int hei, int lr, int gr);
    //WMap(int wid, int hei);
    ~WMap();

    void weightedmap(float *d_imgOutA, float *d_imgOutB, float *d_imgInA, float *d_imgInB, int wid, int hei, int lr, int gr, int gsigma, int guir, double eps);
    void weightedmapTest(float *imgOutA, float *imgOutB, float *imgInA, float *imgInB, int wid, int hei, int lr, int gr, int gsigma, int guir, double eps);

    void laplacianAbsTest(float *imgOut, float *imgIn, int wid, int hei, int lr);
    void gaussianTest(float *imgOut, float *imgIn, int wid, int hei, int gr, int gsigma);
    void saliencymapTest(float *imgOut, float *imgIn, int wid, int hei, int lr, int gr, double gsigma);

private:
    void laplacianAbs(float *d_imgOut, float *d_imgIn, int wid, int hei, int lr);
    void gaussian(float *d_imgOut, float *d_imgIn, int wid, int hei, int gr, int gsigma);
    void saliencymap(float *d_imgOut, float *d_imgIn, int wid, int hei, int lr, int gr, double gsigma);
    void comparison(float *d_imgOut, float *d_imgInA, float *d_imgInB, int wid, int hei);

    int lrad_, grad_;
    float *d_lap_, *d_gau_;
    float *d_tempE_, *d_tempF_;
};

#endif //GFFFUSIONFINAL_WEIGHTEDMAP_H
