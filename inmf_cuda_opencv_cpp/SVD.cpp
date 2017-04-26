#include "SVD.h"

void SVDT::SVDcompute(int m, int n, int lda, int ldu, int ldvt, float *A, float *U, float *S, float *VT)
{
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	//copy A U S and VT to device
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float)*lda*n);
	cudaStat2 = cudaMalloc((void**)&d_U, sizeof(float)*ldu*m);
	cudaStat3 = cudaMalloc((void**)&d_S, sizeof(float)*n);
	cudaStat4 = cudaMalloc((void**)&d_VT, sizeof(float)*ldvt*n);
	cudaStat5 = cudaMalloc((void**)&devInfo, sizeof(int));

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	assert(cudaSuccess == cudaStat5);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(float)*lda*n, cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_U, U, sizeof(float)*ldu*m, cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(d_S, S, sizeof(float)*n, cudaMemcpyHostToDevice);
	cudaStat4 = cudaMemcpy(d_VT, VT, sizeof(float)*ldvt*n, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	//query working space of gesvd
	cusolver_status = cusolverDnSgesvd_bufferSize(
		cusolverH,
		m,
		n,
		&lwork);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
	cudaStat3 = cudaMalloc((void**)&r_work, sizeof(float)*lwork);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat3);

	//compute SVD
	cusolver_status = cusolverDnSgesvd(
		cusolverH,
		'A',
		'A',
		m,
		n,
		d_A,
		lda,
		d_S,
		d_U,
		ldu,
		d_VT,
		ldvt,
		d_work,
		lwork,
		r_work,
		devInfo
		);

	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	//check if SVD is good or not
	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(0 == info_gpu);

	//copy U S and VT to host
	cudaStat1 = cudaMemcpy(U, d_U, sizeof(float)*ldu*m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(S, d_S, sizeof(float)*n, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(VT, d_VT, sizeof(float)*ldvt*n, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	if (d_A) cudaFree(d_A);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);
	if (r_work) cudaFree(r_work);

	if (cublasH) cublasDestroy(cublasH);
	if (cusolverH) cusolverDnDestroy(cusolverH);

	cudaDeviceReset();

}


