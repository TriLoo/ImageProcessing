//
// Created by smher on 17-10-6.
//

#ifndef TWOSCALEREC_TWOSCALEREC_H
#define TWOSCALEREC_TWOSCALEREC_H

#include "iostream"
#include "ctime"
#include "vector"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

using namespace std;
using namespace cv;

#define BLOCKSIZE 16

class TwoScaleRec
{
public:
    TwoScaleRec() = default;

    TwoScaleRec(int wid, int hei);
    ~TwoScaleRec();

    void twoscalerec(float *out, float *d_imgInA, float *d_imgInB, int wid, int hei);
    void twoscalerecTest(float *out, vector<float *> &imgInBase, vector<float *> &imgInDetail, int wid, int hei);
private:
    void multadd(float *d_Out, float *inA, float *inB, float *inC, float *inD, int wid, int hei);

    float *d_tempA_, *d_tempB_, *d_tempC_, *d_tempD_;
    float *d_tempT_;
};

#endif //TWOSCALEREC_TWOSCALEREC_H
