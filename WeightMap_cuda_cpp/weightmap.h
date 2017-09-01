//
// Created by smher on 17-8-19.
//

#ifndef BOXFILTER_BOXFILTER_H
#define BOXFILTER_BOXFILTER_H

#include "heads.h"

using namespace std;

class WeightMap
{
public:
    WeightMap() = default;
    ~WeightMap() = default;

    void laplafilter(float *d_imgOut, const float *d_imgIn, const int height, const int width, const float * __restrict__ d_filter, const int rad);
    void gaussfilter(float *d_imgOut, const float *d_imgIn, const int height, const int width, const float * __restrict__ d_filter, const int rad);
private:
};

/*
class WeightMap
{
public:
    WeightMap() = default;
    WeightMap(int r, int c, int ra):row(r), col(c), rad(ra){}
    WeightMap(int r, int c, int ra, float *In, float *Out):row(r), col(c), rad(ra), imgIn(In), imgOut(Out){}
    ~WeightMap();

    void createfilter(const int ID);
    void boxfilter();
    void weightmap();

    void printFilter();

    float *imgIn, *imgOut;
    float *filter = NULL;
    int rad;
    int row, col;
private:
    float *d_filter = NULL;
    float *d_imgIn = NULL, *d_imgOut = NULL;
    //float *d_filter;
};
*/

#endif //BOXFILTER_BOXFILTER_H
