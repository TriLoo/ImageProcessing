#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    //Mat img = imread("./cow.jpg", IMREAD_COLOR);

    /*
    if(!img.data)
    {
        cout << "read image failed ..." << endl;
        return -1;
    }
    */

    //img.convertTo(img, CV_32F, 1.0);

    //imshow("input", img);

    VideoCapture cap(0);

    if(!cap.isOpened())
    {
        cout << "Open camera failed ..." << endl;
        return -1;
    }

    Mat frames;
    Mat edges;
    bool stop = false;

    // declare needed parameters by stylization function
    float sigma_s = 60.0f;
    float sigma_r = 0.45f;

    while(!stop)
    {
        cap >> frames;

        if(frames.empty())
            break;

        // edge Preserveing filter
        //edgePreservingFilter(frames, frames, RECURS_FILTER, sigma_s, sigma_r);   // 0 : RECURS_FILTER; 1 : NORMCONV_FILTER

        // detail enhancing filter
        //detailEnhance(frames, frames, 10, 0.15f);

        // stylization
        cv::stylization(frames, frames, sigma_s, sigma_r);

        imshow("Frame", frames);

        if(waitKey(33) >= 0)
            stop = true;
    }

    return 0;
}