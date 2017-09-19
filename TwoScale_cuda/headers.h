//
// Created by smher on 17-9-19.
//

#ifndef TWOSCALE_HEADERS_H
#define TWOSCALE_HEADERS_H

#include <iostream>
#include <vector>
#include <cassert>
#include <ctime>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cufft.h>

using namespace std;
using namespace cv;

#define INDX(r, c, w) ((r) * (w) + (c))
// process the log funtion
#define cudaCheckError(err) __checkCUDAError(err, __FILE__, __LINE__)

inline void __checkCUDAError(cudaError_t err, const char *file, int line)
{
    if(err != cudaSuccess)
    {
        cout << err << " in " << file << " at " << line << " line" << endl;
        exit(EXIT_FAILURE);
    }
}

#endif //TWOSCALE_HEADERS_H
