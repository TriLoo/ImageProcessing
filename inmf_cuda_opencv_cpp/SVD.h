#include <stdio.h>
#include <ctime>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
//#include <cublas_v2.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//#include "head.h"

using namespace cv;
using namespace std;

class SVDT
{
private:
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	float *d_A = NULL, *d_U = NULL, *d_S = NULL, *d_VT = NULL;
	int *devInfo = NULL;
	float *d_work = NULL;
	float *r_work = NULL;
	int lwork = 0;
	int info_gpu = 0;

public:
    SVDT(const int m, const int n);
    ~SVDT();
	void SVDcompute(int m, int n, int lda, int ldu, int ldvt, const float *A, float *U, float *S, float *VT);
};
