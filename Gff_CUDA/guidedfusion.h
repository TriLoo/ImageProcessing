//
// Created by smher on 17-9-1.
//

#ifndef GFFFUSION_GUIDEDFUSION_H
#define GFFFUSION_GUIDEDFUSION_H

#include "twoscale.h"
#include "weightmap.h"
#include "tworec.h"

// the most bottom class, derived from three base classes:
// the top virtual base class is  BFilter
// TScale : two scale decomposing function
// WMap   : weight map generation function
// TRec   : two scale restore to get the final fusion result
class GFusion: public TScale, public WMap, public TRec
{
public:
    // set the radius of mean filter, laplacian filter, gauss filter
    // set the sigma parameter of gauss filter
    // set the radius of guided filter and its regulation value
    // more details can be found in below initialization list
    //GFusion(int mfr, int lfr, int gfr, int gs, int gufr, int gur):rad(mfr), lap_rad(lfr), gau_rad(fgr), gau_sig(gs), rad_(gufr), eps_(gur)
    GFusion(int mfr, int lfr, int gfr, int gsv, int gur, float gue):TScale(mfr), WMap(lfr, gfr, gsv, gur, gue)
    {
        cout << "Initialize GFusion Object" << endl;
    }

    ~GFusion(){}

    void guidedfusion(float *imgOut, const float * __restrict__ imgInA, const float * __restrict__ imgInB, int wid, int hei);

private:
};

#endif //GFFFUSION_GUIDEDFUSION_H
