#include <iostream>
#include <ctime>
#include <cassert>
#include <vector>
#include <string>
#include <initializer_list>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "guidedfilter.h"

using namespace std;
using namespace cv;

#define KSIZE (Size(31, 31))    // the size of average filter
#define Sigma 5                 // the standard deviation in X direction In Gaussian
#define VideoDir "/home/smher/myOpenCV/GffFusion/1.avi"

// input origin image, and return high pass mat outH, low pass mat outL
void TwoScale(const Mat &inA, Size ksize, Mat &outH, Mat &outL)
{
	// use average filter to get the twoscale decomposing
	blur(inA, outL, ksize);
	outH = inA - outL;
}

// Get the Saliency Map by Laplace and Gaussian Filters
void SaliencyMap(Mat &imgIn, Mat &SMap)
{
	// Do the Laplace filters 3 * 3
	Laplacian(imgIn, SMap, CV_32F, 3);
	// get the absolute values of 'SMap'
	SMap = abs(SMap);

	// Apply Gaussian filter on SMap
	GaussianBlur(SMap, SMap, Size(5, 5), Sigma);
}

// get the weighted Map, Note: initializer_list only supported by C++11, but doesn't support subscripting
// So we choose to use Vector. Now the number is limited to two
void WeightMap(Mat &imgInA, Mat &imgInB, vector<Mat *> &vecA, vector<Mat *> &vecB, vector<Mat *> &vec)
{
	// get the number of input images to be fused
	int noImg = vec.size();
	int row = imgInA.rows;
	int col = imgInA.cols;

	assert(noImg == 2);

	// declare two temp Mat
	Mat imgMapA = Mat::zeros(row, col, CV_32F);
	Mat imgMapB = Mat::zeros(row, col, CV_32F);

	// get the P value by compare each mat
	// vec[0] : imgInA; vec[1] : imgInB.
	for(int i = 0; i < row; ++i)
		for(int j = 0; j < col; ++j)
		{
			if(vec[0]->at<float>(i, j) > vec[1]->at<float>(i, j))
				imgMapA.at<float>(i, j) = 1;
			else
				imgMapB.at<float>(i, j) = 1;
		}

	// use guided filter to get the weighted map, ----  cannot find this function
	*(vecA[0]) = guidedFilter(imgInA, imgMapA, 20, 0.1, CV_32F);
	*(vecA[1]) = guidedFilter(imgInA, imgMapA, 10, 0.01, CV_32F);
	
	*(vecB[0]) = guidedFilter(imgInB, imgMapB, 20, 0.1, CV_32F);
	*(vecB[1]) = guidedFilter(imgInB, imgMapB, 10, 0.01, CV_32F);

	// test for bilateralFilter ----   can find this function 
	//bilateralFilter(imgInA, *(vec[0]), 10, 1, 1);
}

// Now, the number of layer is limited to Two
void TwoScaleRec(vector<Mat *> &vecW, vector<Mat *> &LayerIn, Mat &LayerOut)
{
	LayerOut = vecW[0]->mul(*LayerIn[0]) + vecW[1]->mul(*LayerIn[1]);
}

void gffFusion(Mat &imgA, Mat &imgB, Mat &Res)
{
	int row = imgA.rows;
	int col = imgA.cols;

	cout << "The dimension of Image is : " << row << " * " << col << endl;

	// declare two more Mats to store the two scale result
	Mat outA_B = Mat::zeros(row, col, CV_32F);
	Mat outA_D = Mat::zeros(row, col, CV_32F);

	Mat outB_B = Mat::zeros(row, col, CV_32F);
	Mat outB_D = Mat::zeros(row, col, CV_32F);

	// convert the input image to float
	imgA.convertTo(imgA, CV_32F, 1.0);
	imgB.convertTo(imgB, CV_32F, 1.0);


	// Decomposing the image into tow scale parts
	TwoScale(imgA, KSIZE, outA_D, outA_B);   // outA: the high pass, outB: the low pass
	TwoScale(imgB, KSIZE, outB_D, outB_B);   // outA: the high pass, outB: the low pass
	// test the decomposing results
	outA_B.convertTo(outA_B, CV_8UC1, 1.0);
	outB_B.convertTo(outB_B, CV_8UC1, 1.0);
	imshow("output A", outA_B);
	imshow("output B", outB_B);
	waitKey(0);

	outA_B.convertTo(outA_B, CV_32F, 1.0);
	outB_B.convertTo(outB_B, CV_32F, 1.0);

	// get the  Saliency Mpas of input images
	Mat SalMapA = Mat::zeros(row, col, CV_32F);
	Mat SalMapB = Mat::zeros(row, col, CV_32F);
	SaliencyMap(imgA, SalMapA);
	SaliencyMap(imgB, SalMapB);

	// Get the Weighted Maps based on Saliency Map and Guided Filter
	//void WeightMap(Mat &imgInA, Mat &imgInB, vector<Mat *> &vecA, vector<Mat *> &vecB, vector<Mat *> &vec)
	Mat WeiMapA_B = Mat::zeros(row, col, CV_32F);
	Mat WeiMapA_D = Mat::zeros(row, col, CV_32F);
	Mat WeiMapB_B = Mat::zeros(row, col, CV_32F);
	Mat WeiMapB_D = Mat::zeros(row, col, CV_32F);

	// Test for the Vector<Mat *>
	vector<Mat *> tempPtrA = {&WeiMapA_B, &WeiMapA_D};
	vector<Mat *> tempPtrB = {&WeiMapB_B, &WeiMapB_D};
	vector<Mat *> tempPtrMap = {&SalMapA, &SalMapB};

	WeightMap(imgA, imgB, tempPtrA, tempPtrB, tempPtrMap);
	//WeightMap(imgA, imgB, vector<Mat *>({&WeiMapA_B, &WeiMapA_D}), vector<Mat *>({&WeiMapB_B, &WeiMapB_D}), vector<Mat *>({&SalMapA, &SalMapB}));

	// Get the restored fusion result by combine the base and detail layers
	//void TwoScaleRec(vector<Mat *> &vecW, vector<Mat *> &LayerIn, Mat &LayerOut)
	tempPtrA.clear();
	tempPtrB.clear();
	tempPtrA = {&WeiMapA_B, &WeiMapB_B};
	tempPtrB = {&outA_B, &outB_B};
	Mat LayerBase = Mat::zeros(row, col, CV_32F);
	Mat LayerDetail = Mat::zeros(row, col, CV_32F);

	TwoScaleRec(tempPtrA, tempPtrB, LayerBase);
	//TwoScaleRec(vector<Mat *>({&WeiMapA_B, &WeiMapB_B}), vector<Mat *>({&outA_B, &outB_B}), LayerBase);
	tempPtrA = {&WeiMapA_D, &WeiMapB_D};
	tempPtrB = {{&outA_D, &outB_D}};
	TwoScaleRec(tempPtrA, tempPtrB, LayerDetail);
	//TwoScaleRec({&WeiMapA_D, &WeiMapB_D}, tempPtrB, LayerDetail);    // Cannot work!
	//TwoScaleRec(vector<Mat *>({&WeiMapA_D, &WeiMapB_D}), vector<Mat *>({&outA_D, &outB_D}), LayerDetail);  // Cannot work!

	// Add the base layer and detail layer to obtain the finnal result
	//Mat Result = Mat::zeros(row, col, CV_32F);
	Res = LayerBase + LayerDetail;
}

int main(int argc, char **argv)
{
	/*
	if(argc != 3)
	{
		cerr << "There is not enough image input ..." << endl;
		return -1;
	}
	 */
    string nameA = "/home/smher/myOpenCV/GffFusion/source20_1.tif";
	string nameB = "/home/smher/myOpenCV/GffFusion/source20_2.tif";
	Mat imgA = imread(nameA, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgB = imread(nameB, CV_LOAD_IMAGE_GRAYSCALE);
    /*
	Mat imgA = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgB = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    */
	if(!imgA.data)
	{
		cerr << "Read image A fail ..." << endl;
		return -1;
	}
	if(!imgB.data)
	{
		cerr << "Read image B fail ..." << endl;
		return -1;
	}

	int row = imgA.rows;
	int col = imgA.cols;

	double duration = 0;
	clock_t start, end;
	
	// start timing
	start = clock();

	VideoCapture capture(VideoDir);
	// do the fusion by calling the gffFusion function
	// declare two more Mats to store the two scale result
	Mat outA_B = Mat::zeros(row, col, CV_32F);
	Mat outA_D = Mat::zeros(row, col, CV_32F);

	Mat outB_B = Mat::zeros(row, col, CV_32F);
	Mat outB_D = Mat::zeros(row, col, CV_32F);

	// convert the input image to float
	imgA.convertTo(imgA, CV_32F, 1.0);
	imgB.convertTo(imgB, CV_32F, 1.0);


	// Decomposing the image into tow scale parts
	TwoScale(imgA, KSIZE, outA_D, outA_B);   // outA: the high pass, outB: the low pass
	TwoScale(imgB, KSIZE, outB_D, outB_B);   // outA: the high pass, outB: the low pass
	// test the decomposing results
	outA_B.convertTo(outA_B, CV_8UC1, 1.0);
	outB_B.convertTo(outB_B, CV_8UC1, 1.0);
	imshow("output A", outA_B);
	imshow("output B", outB_B);

	outA_B.convertTo(outA_B, CV_32F, 1.0);
	outB_B.convertTo(outB_B, CV_32F, 1.0);

	// get the  Saliency Mpas of input images
	Mat SalMapA = Mat::zeros(row, col, CV_32F);
	Mat SalMapB = Mat::zeros(row, col, CV_32F);
	SaliencyMap(imgA, SalMapA);
	SaliencyMap(imgB, SalMapB);

	// Get the Weighted Maps based on Saliency Map and Guided Filter
	//void WeightMap(Mat &imgInA, Mat &imgInB, vector<Mat *> &vecA, vector<Mat *> &vecB, vector<Mat *> &vec)
	Mat WeiMapA_B = Mat::zeros(row, col, CV_32F);
	Mat WeiMapA_D = Mat::zeros(row, col, CV_32F);
	Mat WeiMapB_B = Mat::zeros(row, col, CV_32F);
	Mat WeiMapB_D = Mat::zeros(row, col, CV_32F);

	// Test for the Vector<Mat *>
	vector<Mat *> tempPtrA = {&WeiMapA_B, &WeiMapA_D};
	vector<Mat *> tempPtrB = {&WeiMapB_B, &WeiMapB_D};
	vector<Mat *> tempPtrMap = {&SalMapA, &SalMapB};

	WeightMap(imgA, imgB, tempPtrA, tempPtrB, tempPtrMap);
	//WeightMap(imgA, imgB, vector<Mat *>({&WeiMapA_B, &WeiMapA_D}), vector<Mat *>({&WeiMapB_B, &WeiMapB_D}), vector<Mat *>({&SalMapA, &SalMapB}));

	// Get the restored fusion result by combine the base and detail layers
	//void TwoScaleRec(vector<Mat *> &vecW, vector<Mat *> &LayerIn, Mat &LayerOut)
	tempPtrA.clear();
	tempPtrB.clear();
	tempPtrA = {&WeiMapA_B, &WeiMapB_B};
	tempPtrB = {&outA_B, &outB_B};
	Mat LayerBase = Mat::zeros(row, col, CV_32F);
	Mat LayerDetail = Mat::zeros(row, col, CV_32F);

	TwoScaleRec(tempPtrA, tempPtrB, LayerBase);
	//TwoScaleRec(vector<Mat *>({&WeiMapA_B, &WeiMapB_B}), vector<Mat *>({&outA_B, &outB_B}), LayerBase);
	tempPtrA = {&WeiMapA_D, &WeiMapB_D};
	tempPtrB = {{&outA_D, &outB_D}};
	TwoScaleRec(tempPtrA, tempPtrB, LayerDetail);
	//TwoScaleRec({&WeiMapA_D, &WeiMapB_D}, tempPtrB, LayerDetail);    // Cannot work!
	//TwoScaleRec(vector<Mat *>({&WeiMapA_D, &WeiMapB_D}), vector<Mat *>({&outA_D, &outB_D}), LayerDetail);  // Cannot work!

	// Add the base layer and detail layer to obtain the finnal result
	//Mat Result = Mat::zeros(row, col, CV_32F);
	Mat Result = Mat::zeros(row, col, CV_32F);           // store the fusion result
	Result = LayerBase + LayerDetail;
	//gffFusion(imgA, imgB, Result);


	// stop timing
	end = clock();
	//duration = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000;
	duration = double(end - start) / CLOCKS_PER_SEC * 1000;
	cout << "The total time is : " << duration << " ms" << endl;

	// display the input image
	imgA.convertTo(imgA, CV_8UC1, 1.0);
	imshow("image A", imgA);

	imgB.convertTo(imgB, CV_8UC1, 1.0);
	imshow("image B", imgB);

	Result.convertTo(Result, CV_8UC1, 1.0);
	imwrite("Result.png", Result);
	imshow("Fusion Result", Result);

	waitKey(0);

	return 0;
}

