//
// Created by smher on 17-11-18.
//

#ifndef GFFFUSIONFINAL_HEADERS_H
#define GFFFUSIONFINAL_HEADERS_H

#include "iostream"
#include "vector"
#include "stdexcept"
#include "cassert"
#include "ctime"
//#include "chrono"
//#include "boost/thread.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cufft.h"
#include "nppi.h"

using namespace std;
using namespace cv;

#define BLOCKSIZE 16
#define INDX(r, c, w) ((r) * (w) + (c))

#define cudaCheckErrors(err) __cudaCheckErrors(err, __FILE__, __LINE__)

inline void __cudaCheckErrors(cudaError_t err, const char *file, const int line)
{
    if(err != cudaSuccess)
        cerr << cudaGetErrorString(err) << " : In " << file << ". At " << line << " line." << endl;
}


#endif //GFFFUSIONFINAL_HEADERS_H
