/*
 * twoscale.h
 *
 *  Created on: Sep 20, 2017
 *      Author: smher
 */

#ifndef TWOSCALE_H_
#define TWOSCALE_H_

#include <iostream>
#include <cassert>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class BFilter
{
public:
	BFilter(int wid, int hei);
	~BFilter();
	void boxfilter(float *d_imgOut, float *d_imgIn, int wid, int hei, int filterR);
	void boxfilterTest(float *imgOut, float *imgIn, int wid, int hei, int filterR);
private:
	float *d_imgIn_, *d_imgOut_;
};

class TScale: public virtual BFilter
{
public:
	TScale(int w, int h):BFilter(w, h){}
	~TScale(){}

	void twoscale(float *d_imgOutA, float *d_imgOutB, float *d_imgIn, int wid, int hei, int filterR);
	void twoscaleTest(float *imgOutA, float *imgOutB, float *imgIn, int wid, int hei, int filterR);
private:
};




#endif /* TWOSCALE_H_ */
