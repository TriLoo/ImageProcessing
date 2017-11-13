//
// Created by smher on 17-11-11.
//

#ifndef HISTEQUNPP_HISTEQU_H
#define HISTEQUNPP_HISTEQU_H

#include "iostream"
#include "vector"
//#include "boost/chrono"
#include "stdexcept"
#include "cassert"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "npp.h"
#include "nppi.h"

using namespace std;
using namespace cv;

//#ifdef __cplusplus
//extern "C"
//{
class histEqu
{
public:
    int histEquNpp(Mat &imgOut, Mat &imgIn);
};
//};
//#endif

#endif //HISTEQUNPP_HISTEQU_H
