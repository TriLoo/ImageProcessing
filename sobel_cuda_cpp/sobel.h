/*
 * soble.h
 *
 *  Created on: May 23, 2017
 *      Author: smher
 */

#ifndef SOBLE_H_
#define SOBLE_H_

#include <iostream>
#include <ctime>
#include <cmath>
#include <assert.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

class mySobel
{
public:
	mySobel() = default;
	mySobel(int m, int n, float *A):row(m), col(n), img_H(A){}
	void SobelCompute(float *A, float *B);
private:
	cudaError_t cudaState = cudaSuccess;
	int row = 0;
	int col = 0;
	float *img_H = NULL;
	//float *img_D = nullptr;
};



#endif /* SOBLE_H_ */
