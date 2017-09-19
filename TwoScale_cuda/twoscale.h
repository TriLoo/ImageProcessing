//
// Created by smher on 17-9-19.
//

#ifndef TWOSCALE_TWOSCALE_H
#define TWOSCALE_TWOSCALE_H

#include "headers.h"
#include "filter.h"

class TwoScale: public virtual Filter
{
public:
    TwoScale() = default;
    TwoScale(int r, int w, int h):Filter(r, w, h){}
    ~TwoScale();

    void twoscale(float *d_imgOut, float *d_imgIn, int wid, int hei, int rad);
private:
};

#endif //TWOSCALE_TWOSCALE_H
