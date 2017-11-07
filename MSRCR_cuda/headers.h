//
// Created by smher on 17-11-7.
//

#ifndef MSRCR_HEADERS_H
#define MSRCR_HEADERS_H

#include "iostream"
#include "vector"
#include "stdexcept"
#include "cassert"
#include "ctime"
//#include "chrono"
#include "boost/thread.hpp"
#include "boost/format.hpp"
#include "boost/chrono.hpp"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
//#include "opencv2/"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cufft.h"

using namespace std;
using namespace cv;

#define cudaCheckErrors(err) __cudaCheckErrors(err, __FILE__, __LINE__)

inline void __cudaCheckErrors(cudaError_t err, const char *file, const int line)
{
    cerr << "Error at " << file << ", line : " << line << endl;
    cerr << "err.what() : " << err << endl;
}

#endif //MSRCR_HEADERS_H
