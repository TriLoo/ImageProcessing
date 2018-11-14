#include "SVD.h"

SVDT::SVDT(const int m, const int n)
{
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cublas_status = cublasCreate(&cublasH);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);

	cudaStat1 = cudaMalloc((void **)&d_A, sizeof(float) * m * n);   // Storing input image
    cudaStat2 = cudaMalloc((void **)&d_U, sizeof(float) * m * m);    // Left singular matrix
	cudaStat3 = cudaMalloc((void **)&d_S, sizeof(float) * m * n);   // singular - values
	cudaStat4 = cudaMalloc((void **)&d_VT, sizeof(float) * n * n);
	cudaStat5 = cudaMalloc((void**)&devInfo, sizeof(int));

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	assert(cudaSuccess == cudaStat5);

	cusolver_status = cusolverDnSgesvd_bufferSize(
			cusolverH,
			m,
			n,
			&lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
	cudaStat3 = cudaMalloc((void**)&r_work, sizeof(float)*lwork);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat3);
}

SVDT::~SVDT()
{
	//cusolver_status = cusolverDnDestroy(cusolverH);
	//assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    if (d_A)
		cudaFree(d_A);
	if (d_U)
		cudaFree(d_U);
	if (d_S)
		cudaFree(d_S);
	if (d_VT)
		cudaFree(d_VT);
	if (devInfo)
		cudaFree(devInfo);
	if (d_work)
		cudaFree(d_work);
	if (r_work)
		cudaFree(r_work);

	if (cublasH)
		cublasDestroy(cublasH);
	if (cusolverH)
		cusolverDnDestroy(cusolverH);
}

void SVDT::SVDcompute(int m, int n, int lda, int ldu, int ldvt, const float *A, float *U, float *S, float *VT)
{
	cudaStat1 = cudaMemcpy(d_A, A, sizeof(float)*lda*n, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
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
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	cout << "SVD Time: " << elapsedTime << " ms." << endl;

	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

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
	//cudaDeviceReset();
}
