//
// Created by smher on 18-12-9.
//

#ifndef RDLW_SAL_FUSIONCUDA_HEADERS_H
#define RDLW_SAL_FUSIONCUDA_HEADERS_H

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__)
#define INDX(r, c, w) ((r) * (w) + (c))

void __cudaCheckError(cudaError_t err, const char *filename, const int line);
int iDiv(int a, int b);

namespace IVFusion
{
    const int BLOCKSIZE = 16;         // only internal linkage, non-const variables have external linkage.
    /**
      // leading to multiple difinition of __cudaCheckError & iDiv.
      // see:
      //     https://stackoverflow.com/questions/40514928/multiple-definition-when-using-namespace-in-header-file?rq=1
     void __cudaCheckError(cudaError_t err, const char *filename, const int line)
    {
        if(err != cudaSuccess)
        {
            std::cout << cudaGetErrorString(err) << ": in " << filename << ". At " << line << " line." << std::endl;
        }
    }

    int iDiv(int a, int b)
    {
        if(a % b == 0)
            return a / b;
        else
            return a / b + 1;
    }
    */
}

#endif //RDLW_SAL_FUSIONCUDA_HEADERS_H
