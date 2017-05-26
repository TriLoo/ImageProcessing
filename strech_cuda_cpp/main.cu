#include <iostream>
#include <assert.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

typedef unsigned char uchar;
typedef unsigned int uint;
/*
using uchar = std::unsigned char; // Not work, namespace-qualified name is required... and below same
using uint = std::unsigned int;
*/

// declare the texture reference
texture<float, cudaTextureType2D, cudaReadModeElementType> texIn;

__global__ void StrechKernel(float *A, float *B, int row, int col, float max, float min)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

	if( x < row && y < col)
	{
		float minus = max - min;
		B[offset] = ((A[offset] - min) / minus) * 255;
	}
}

int main(int argc, char **argv)
{
	/*
	if(argc != 2)
	{
		cerr << "no image name input ..." << endl;
		return -1;
	}
*/
	//Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img = imread("LenaWithoutContrast.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	if(!img.data)
	{
		cerr << "read image failed ..." << endl;
		return -1;
	}
	else
	{
		cout << "read image success: " << img.rows << " * " << img.cols << endl;
	}

	int row = img.rows;
	int col = img.cols;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	// find the minimum and maximum value of input image
	uint *hist = new uint [256]();

	const int SIZE = row * col * sizeof(float);

	uchar val = 0;
	for(int i = 0; i < row; i++)
		for(int j = 0; j < col; j++)
		{
			val = img.at<uchar>(i, j);
			hist[val]++;
		}
	// get the max and min value of input image
	float Max, Min;
	for(int i = 0; i < 256; i++)
	{
		Min = i;
		if(hist[i] != 0)
			break;
	}
	for(int i = 255; i >= 0; i--)
	{
		Max = i;
		if(hist[i] != 0)
			break;
	}

	cout << "The maximum value is : " << Max << " And the minimum value is : " << Min << endl;

	// prepare for kernel function
	dim3 block(16, 16);   // every block has 16 * 16 threads
	dim3 grid((row + 15)/16, (col + block.y)/16);

	cudaError_t cudaState = cudaSuccess;

	// calculate the strech part
	// change the image to float version
	img.convertTo(img, CV_32F, 1.0);
	float *imgIn = (float *)(img.data);
	float *imgIn_dev;

	cudaState = cudaMalloc((void **)&imgIn_dev, SIZE);
	assert(cudaState == cudaSuccess);
	cudaState = cudaMemcpy(imgIn_dev, imgIn, SIZE, cudaMemcpyHostToDevice);
	assert(cudaState == cudaSuccess);
	float *imgOut_dev;
	cudaState = cudaMalloc((void **)&imgOut_dev, SIZE);
	assert(cudaState == cudaSuccess);
	cudaState = cudaMemset(imgOut_dev, 0, SIZE);
	assert(cudaState == cudaSuccess);

	StrechKernel<<<grid, block>>>(imgIn_dev, imgOut_dev, row, col, Max, Min);
	// copy the data back to host
	float *imgOut = new float [row * col];
	cudaState = cudaMemcpy(imgOut, imgOut_dev, SIZE, cudaMemcpyDeviceToHost);
	assert(cudaState == cudaSuccess);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);               // very important, if not synchronize the event, then you should get the wrong value !!!
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "GPU used time is : " << elapsedTime << " ms" << endl;

	Mat Res(row, col, CV_32F, imgOut, 0);
	Res.convertTo(Res, CV_8UC1, 1.0);

	imwrite("result.jpg", Res);
	imshow("result", Res);

	waitKey(0);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	delete [] hist;
	delete [] imgOut;

	cudaFree(imgIn_dev);
	cudaFree(imgOut_dev);

	return 0;
}
