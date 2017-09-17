#ifndef HEADERS_
#define HEADERS_

#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cufft.h>

#include <driver_types.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

/*
class TimeCal
{
public:
    TimeCal()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void StartTimer()
    {
        cudaEventRecord(start, 0);
    }

    void StartStop()
    {
        cudaEventSynchronize(stop);
        cudaEventRecord(stop, 0);
    }

    void DurationTime()
    {
        cout << "GPU time :" << cudaEventElapsedTime(0, start, stop);
    }
private:
    cudaEvent_t start, stop;
};
*/

#endif
