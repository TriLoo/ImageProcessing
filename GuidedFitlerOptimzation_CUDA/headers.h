//
// Created by smher on 18-1-16.
//

#ifndef GUIDEDFILTEROPTIMIZE_HEADERS_H
#define GUIDEDFILTEROPTIMIZE_HEADERS_H

#include "iostream"
#include "vector"
#include "stdexcept"
#include "cassert"
#include "memory"
#include "algorithm"
#include "functional"
#include "chrono"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "npp.h"
#include "nppcore.h"
#include "nppi.h"

#define INDX(r, c, w) ((r) * (w) + (c))

#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__)

inline
void __cudaCheckError(cudaError_t err, const char* filename, const int line)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << ": In " << filename << ". At" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif //GUIDEDFILTEROPTIMIZE_HEADERS_H
