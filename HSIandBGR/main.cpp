#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// Input: 8UC3
// Output: 32FC1
void convertBGRtoHSI(cv::Mat& imgI, cv::Mat& imgS, cv::Mat& imgH, const Mat& imgIn)
{
    float r, g, b, h, s, ic;
	for(int i = 0; i < imgIn.rows; ++i)
	{
		for(int j = 0; j < imgIn.cols; ++j)
        {
            Vec3f pixel = imgIn.at<Vec3f>(i, j);
            b = pixel[0];
            g = pixel[1];
            r = pixel[2];

            ic = (b + g + r) / 3.0;

            float min_val = 0;
            min_val = std::min(r, std::min(b, g));
            s = 1 - 3 * (min_val / (r + b + g));
            if(s < 0.00001)
                s = 0;
            else if(s > 0.99999)
                s = 1;

            if(s > 0.00001)
            {
                h = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g) * (r - g)) + ((r - b) * (g - b)));
                h = acos(h); // h -> radians

                if(b <= g)
                    h = h;
                else
					h = 2 * M_PI - h;
            }
            else
                h = 0;

            imgI.at<float>(i, j) = ic;
            imgS.at<float>(i, j) = s;
            imgH.at<float>(i, j) = h;   // Now h range [0, 2pi]  
			/**  // conversion from RGB to HSI, see bookmarks from firefox
            imgI.at<float>(i, j) = ic * 255;
            imgS.at<float>(i, j) = s * 100;
            imgH.at<float>(i, j) = (h * 180) / M_PI;   
			*/
        }
	}
	//normalize(imgI, imgI, 0.0, 1.0, NORM_MINMAX);
}

// Inputs: CV_32FC1
// Output: CV_32FC3
void convertHSItoBGR(Mat& imgOut, const cv::Mat &imgI, const cv::Mat& imgS, const cv::Mat& imgH)
{
	int rows = imgI.rows, cols = imgI.cols;
	float h, s, ic, r, g, b;
	for(int i = 0; i < rows; ++i)
	{
		for(int j = 0; j < cols; ++j)
		{
			h = imgH.at<float>(i, j);        // [0, 2*PI]
			s = imgS.at<float>(i, j);
			ic = imgI.at<float>(i, j);

			if(h < 2 * M_PI / 3)   // RG
			{
				b = ic * (1 - s);
				r = ic * (1 + s * cos(h) / cos(M_PI / 3 - h));
				g = 3 * ic - (r + b);
			}
			else if(h < 4 * M_PI / 3)   // GB
			{
				h -= 2 * M_PI / 3;
				r = ic * (1 - s);
				g = ic * (1 + s * cos(h) / cos(M_PI / 3 - h));
				b = 3 * ic - (r + g);
			}
			else                        // BR
			{
				h -= 4 * M_PI / 3;
				g = ic * (1 - s);
				b = ic * (1 + s * cos(h) / cos(M_PI / 3 - h));
				r = 3 * ic - (g + b);
			}

			//imgOut.at<Vec3f>(i, j) = Vec3f(b, g, r);
			imgOut.at<Vec3f>(i, j)[0] = b;
			imgOut.at<Vec3f>(i, j)[1] = g;
			imgOut.at<Vec3f>(i, j)[2] = r;
		}
	}
}

int main()
{
	Mat img = imread("lena.jpg", IMREAD_COLOR);
	//cout << IMREAD_COLOR << endl;
	//Mat img = imread("baboon.jpg", IMREAD_COLOR);
	if(img.empty())
	{
		cout << "Read image failed." << endl;
		exit(1);
	}

	img.convertTo(img, CV_32FC3, 1.0/255);

	cout << "Read image successfully." << endl;

	Mat imgOut = Mat::zeros(img.size(), CV_32FC3);
    //Mat imgOut;

	Mat imgI = Mat::zeros(img.size(), CV_32FC1);
	Mat imgS = Mat::zeros(img.size(), CV_32FC1);
	Mat imgH = Mat::zeros(img.size(), CV_32FC1);

	convertBGRtoHSI(imgI, imgS, imgH, img);

	/*
    vector<Mat> toMerge = {imgH, imgS, imgI};
    merge(toMerge, imgOut);

	imgOut.convertTo(imgOut, CV_8UC3, 1);
	imshow("test", imgOut);
	waitKey(0);
	*/

	/**
    normalize(imgOut, imgOut, 0.0, 1.0, NORM_MINMAX);
    imgOut.convertTo(imgOut, CV_8UC3, 255);
    imwrite("HSIImage.png", imgOut);
	*/

	// for test convert BGR to HSI
	//imshow("I-Channel", imgI);
	//waitKey(0);

	convertHSItoBGR(imgOut, imgI, imgS, imgH);
    //cout << imgOut.rowRange(0, 10).colRange(0, 10) << endl;

	//imshow("Input", img);
	//imshow("Output", imgOut);
    //normalize(imgOut, imgOut, 0.0, 1.0, NORM_MINMAX);
    imgOut.convertTo(imgOut, CV_8UC3, 255);
	imshow("Restored", imgOut);
    //imwrite("Restored.png", imgOut);

	waitKey(0);

	return 0;
}


