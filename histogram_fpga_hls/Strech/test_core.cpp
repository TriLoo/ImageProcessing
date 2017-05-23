#include <stdio.h>
#include "core.h"
#include <opencv2/core/core.hpp>
#include <hls_opencv.h>

// Image File path
#define INPUT_IMAGE_CORE "/home/smher/VivadoHLS/HLS_Strech/lena.bmp"
#define OUTPUT_IMAGE_CORE "/home/smher/VivadoHLS/HLS_Strech/lenaWithContrast.bmp"
char outImage[320][240];

void saveImage(const std::string path, cv::InputArray inArr)
{
	double min;
	double max;
	cv::minMaxIdx(inArr, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(inArr, adjMap, 255 / max);
	cv::imwrite(path,adjMap);
}

int main()
{
	// Read input image
	printf("Load image %s\n",INPUT_IMAGE_CORE);
	cv::Mat imageSrc;
	imageSrc = cv::imread(INPUT_IMAGE_CORE);
	// Convert to grayscale
	cv::cvtColor(imageSrc, imageSrc, CV_BGR2GRAY);
	printf("Image Rows:%d Cols:%d\n",imageSrc.rows, imageSrc.cols);

	// Define streams for input and output
	hls::stream<uint_8_side_channel> inputStream;
	hls::stream<uint_8_side_channel> outputStream;

	// OpenCV mat that point to a array (cv::Size(Width, Height))
	cv::Mat imgCvOut(cv::Size(imageSrc.cols, imageSrc.rows), CV_8UC1, outImage, cv::Mat::AUTO_STEP);

	// Populate the input stream with the image bytes
	for (int idxRows=0; idxRows < imageSrc.rows; idxRows++)
	{
		for (int idxCols=0; idxCols < imageSrc.cols; idxCols++)
		{
			uint_8_side_channel valIn;
			valIn.data = imageSrc.at<unsigned char>(idxRows,idxCols);
			valIn.keep = 1; valIn.strb = 1; valIn.user = 1; valIn.id = 0; valIn.dest = 0;
			inputStream << valIn;
		}
	}

	// Min and max of the input image were calculated in matlab
	doHistStrech(inputStream, outputStream, 40, 111);

	// Take data from the output stream to our array outImage (Pointed in opencv)
	for (int idxRows=0; idxRows < imageSrc.rows; idxRows++)
	{
		for (int idxCols=0; idxCols < imageSrc.cols; idxCols++)
		{
			uint_8_side_channel valOut;
			outputStream.read(valOut);
			outImage[idxRows][idxCols] = valOut.data;
		}
	}

	// Save image out file
	saveImage(std::string(OUTPUT_IMAGE_CORE) ,imgCvOut);
	return 0;
}
/*
#include <stdio.h>
#include "core.h"
#include <opencv2/core/core.hpp>
#include <hls_opencv.h>

// Image file path

char outImage[320][240];

void saveImage(const std::string path, cv::InputArray inArr)
{
	double min;
	double max;
	cv::minMaxIdx(inArr, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(inArr, adjMap, 255/max);
	cv::imwrite(path, adjMap);
}

int main(void)
{
	char *inputImg = "/home/smher/VivadoHLS/HLS_Strech/lena.bmp";
	char *outputImg = "/home/smher/VivadoHLS/HLS_Strech/lenaWithContrast.bmp";
	// Read input image
	printf("load image : %s \n", inputImg);
	cv::Mat imageSrc;
	imageSrc = cv::imread(inputImg);

	// convert the image to grayscale
	cv::cvtColor(imageSrc, imageSrc, CV_BGR2GRAY);
	printf("Image rows: %d, Cols: %d\n", imageSrc.rows, imageSrc.cols);

	// Define streams for input and output
	hls::stream<uint_8_side_channel> inputStream;
	hls::stream<uint_8_side_channel> outputStream;

	// OpenCV mat that point to a array(cv::Size(width, height))
	cv::Mat imgCvOut(cv::Size(imageSrc.cols, imageSrc.rows), CV_8UC1, outImage, cv::Mat::AUTO_STEP);

	// Popular the input stream with the image bytes
	for(int idxRows = 0; idxRows < imageSrc.rows; idxRows++)
	{
		for(int idxCols = 0; idxCols < imageSrc.cols; idxCols++)
		{
			uint_8_side_channel valIn;
			valIn.data = imageSrc.at<unsigned char>(idxRows, idxCols);
			valIn.keep = 1;
			valIn.strb = 1;
			valIn.user = 1;
			valIn.id = 0;
			valIn.dest = 0;
			inputStream << valIn;
		}
	}

	// Min and Max of the input image were calculated in Matlab
	doHistStrech(inputStream, outputStream, 0, 255);

	// take data from the output stream to our array outImage(Pointed in OpenCV)
	for(int idxRows = 0; idxRows < imageSrc.rows; idxRows++)
	{
		for(int idxCols = 0; idxCols < imageSrc.cols; idxCols++)
		{
			uint_8_side_channel valOut;
			outputStream.read(valOut);
			outImage[idxRows][idxCols] = valOut.data;
		}
	}

	// save image out file
	saveImage(outputImg, imgCvOut);
	return 0;
}
*/
