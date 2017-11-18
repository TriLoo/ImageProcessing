//
// Created by smher on 17-11-18.
//

#ifndef GFFFUSIONFINAL_TWOSCALE_H
#define GFFFUSIONFINAL_TWOSCALE_H

#include "BFilter.h"

class TwoScale: public virtual BFilter
{
public:
    TwoScale() = default;
    ~TwoScale();    // is also a virtual function

    void twoscale(float *d_imgOutA, float *d_imgOutB, float *d_imgIn,const int wid, const int hei, const int filterR);
    void twoscaleTest(float *imgOutA, float *imgOutB, float *imgIn,const int wid, const int hei, const int filterR);

private:
};

#endif //GFFFUSIONFINAL_TWOSCALE_H
