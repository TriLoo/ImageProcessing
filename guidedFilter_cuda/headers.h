//
// Created by smher on 17-9-29.
//

#ifndef GUIDEDFILTER_HEADERS_H
#define GUIDEDFILTER_HEADERS_H

#include <iostream>
#include <vector>
#include "ctime"
#include <cassert>
#include "stdexcept"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/edge_filter.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;
using namespace cv;

#define BLOCKSIZE 16

#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__)

inline void __cudaCheckError(cudaError_t err, const char *file, int line)
{
    if(err != cudaSuccess)
    {
        cout << err << " in " << file << " at " << line << " line." << endl;
    }
}

#endif //GUIDEDFILTER_HEADERS_H
