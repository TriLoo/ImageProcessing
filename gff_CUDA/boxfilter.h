//
// Created by smher on 17-8-3.
//

#ifndef BOXFILTER_BOXFILTER_H
#define BOXFILTER_BOXFILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

using namespace std;

class BFilter
{
public:
	BFilter() = default;
	~BFilter();

	void boxfilter();
	void print();

	int width, height, rad;
	float *data, *dataInD, *dataOutD;
private:
};

#endif //BOXFILTER_BOXFILTER_H
