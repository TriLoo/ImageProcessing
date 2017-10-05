//
// Created by smher on 17-10-3.
//

#ifndef WEIGHTEDMAP_HEADERS_H
#define WEIGHTEDMAP_HEADERS_H

#include "iostream"
#include "vector"
#include "ctime"
#include "cassert"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

using namespace std;
using namespace cv;

#define BLOCKSIZE 16

#define cudaCheckErrors(err) __checkCUDAError(err, __FILE__, __LINE__)

inline void __checkCUDAError(cudaError_t err, const char *file, int line)
{
    if(err != cudaSuccess)
        cout << err << " in " << file << " at " << line << " line." << endl;
}

#endif //WEIGHTEDMAP_HEADERS_H
