//
// Created by smher on 17-9-1.
//

#ifndef GFFFUSION_GUIDEDFILTER_H
#define GFFFUSION_GUIDEDFILTER_H

#include "boxfilter.h"

// Caution
// share BFilter to another class : TScale
// so, set BFilter to virtual base class
class GFilter : public virtual BFilter
{
public:
    GFilter(float e, int r):eps_(e), rad_(r)
    {
        cout << "In Guided filter" << endl;
    }
    ~GFilter()
    {
        cudaFree(d_guifilter_);
    }

    void createGufilter();
    // d_inI : the guidance image, d_inP : the filtering image
    void guidedfilter(float *d_out, float *d_inI, float *d_inP, int wid, int hei);

private:
    float eps_;    // the regular element
    float rad_;    // the radius of guided filter
    float *d_guifilter_;   // the mean filter
};

#endif //GFFFUSION_GUIDEDFILTER_H
