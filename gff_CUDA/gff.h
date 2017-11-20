//
// Created by smher on 17-11-18.
//

#ifndef GFFFUSIONFINAL_GFF_H
#define GFFFUSIONFINAL_GFF_H

//#include "headers.h"
#include "TwoScale.h"
#include "WeightedMap.h"

class GFF : public TwoScale, public WMap
{
public:
    GFF() = default;
    GFF(int wid, int hei, int lr, int gr);
    ~GFF();

    // tsr : tow scale radius
    // gr  : gaussian filter radius
    // gsig : gaussian filter sigma
    // guir : guided filter radius
    // guieps : guided filter eps
    void gffFusion(float *d_imgOut, float *d_imgInA, float *d_imgInB, int wid, int hei, int tsr, int lr, int gr, double gsig,
                        int guir, double guieps);
    void gffFusionTest(float *imgOut, float *imgInA, float *imgInB, int wid, int hei, int tsr, int lr, int gr, double gsig,
                   int guir, double guieps);
    void MergeTest(float *imgOut, vector<float *> &imgInS, vector<float *> &wampInS, int wid, int hei);
private:
    void Merge(float *d_imgOut, vector<float *>& d_imgInS, vector<float *>& d_wmapInS,const int wid, const int hei);
};

#endif //GFFFUSIONFINAL_GFF_H
