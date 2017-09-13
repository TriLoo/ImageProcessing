/*
 * boxfilter.h
 *
 *  Created on: Sep 13, 2017
 *      Author: smher
 *  Description: this file include the four kinds implementation of boxfilter, three are based on convolution but different from the used memory: including, global memory,
 *  				shared memory, texture memory.  The last kind boxfilter is based on the normal approach using separable add & minus.
 */

#ifndef BOXFILTER_H_
#define BOXFILTER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cassert>

using namespace std;

class BFilter
{
public:
	BFilter(int wid, int hei, int filterW);
	~BFilter();

	void boxfilterGlo(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW);
	void boxfilterSha(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW);
	void boxfilterTex(float *imgOut, float *imgIn, int wid, int hei, float *filter, int filterW);
	void boxfilterSep(float *imgOut, float *imgIn, int wid, int hei, int filterW);       // only support mean filtering

private:
	float *d_imgIn_, *d_imgOut_, *d_filter_;
	float *d_imgIn_Pitch_;
};



#endif /* BOXFILTER_H_ */
