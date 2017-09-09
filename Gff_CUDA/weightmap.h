//
// Created by smher on 17-9-1.
//

#ifndef GFFFUSION_WEIGHTMAP_H
#define GFFFUSION_WEIGHTMAP_H

//#include "header.h"
//#include "boxfilter.h"
#include "guidedfilter.h"

class WMap : public GFilter
{
public:
    // gur : guided filter radius, gue : guided filter eps
    WMap(int lr, int gr, int gs, int gur, float gue):lap_rad(lr), gau_rad(gr), gau_sig(gs), GFilter(gue, gur)
    {
        //cout << "In Weight map" << endl;
        //cudaMalloc((void **)&d_lap_filter_, sizeof(float) * (2 * lr + 1) * (2 * lr + 1));
        //cout << "Lap Size on Device = " << ((2 * lr + 1) * (2 * lr + 1)) << endl;
        cudaMalloc((void **)&d_gau_filter_, sizeof(float) * (2 * gr + 1) * (2 * gr + 1));
        cout << "WMap initialized Success" << endl;
    }
    ~WMap()
    {
        cudaFree(d_lap_filter_);
        cudaFree(d_gau_filter_);
    }

    // create laplace & guass filters on device memory
    void createfilterLG();

    // use laplace & gauss filter to generate saliency map
    // laplace filter based on boxfilter
    // the input image is stored on global memory
    void laplafilter(float *d_out, float *d_in, int wid, int hei);
    // TODO: the input image is stored on 2D Pitch memory and do laplacian filter on it
    //void laplafilter(float *d_out, float *d_in, size_t pitch, int wid, int hei);
    // gauss filter based on boxfilter
    void gaussfilter(float *d_out, float *d_in, int wid, int hei);

    // compare the two saliency map to generate weight map
    void weightmap(float *d_outA, float *d_outB, float *d_inA, float *d_inB, int wid, int hei);
    // use the weight map and guided filter to generate the final result map
    void resultmap(float *d_out, float *d_inI, float *d_inP, int wid, int hei, int rad, float eps);
private:
    int lap_rad, gau_rad, gau_sig;    // gau_sig : value of gauss sigma
    //int gui_rad, gui_eps;                 // gui_rad : guided filter radius, gui_eps : guided eps
    float *d_lap_filter_;
    float *d_gau_filter_;
};

#endif //GFFFUSION_WEIGHTMAP_H

