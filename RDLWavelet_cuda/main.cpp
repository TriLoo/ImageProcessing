/**
 * @author smh
 * @date 2018.12.06
 *
 */
#include "headers.h"
#include "RDLWavelet.h"

using namespace std;
using namespace cv;

int main() {
    //std::cout << "Hello, World!" << std::endl;
    //Mat imgIn = imread("lena.jpg", IMREAD_GRAYSCALE);
    Mat imgIn = imread("Kaptein_1654_IR.bmp", IMREAD_GRAYSCALE);

    if(imgIn.empty())
    {
        cout << "Read image failed." << endl;
        return -1;
    }

    if(imgIn.type() != CV_32F)
        imgIn.convertTo(imgIn, CV_32F, 1.0/255);
    assert(imgIn.isContinuous());

    int rows = imgIn.rows;
    int cols = imgIn.cols;

    vector<Mat> imgOuts(4, Mat::zeros(Size(cols, rows), CV_32FC1));
    Mat imgRestore = Mat::zeros(Size(cols, rows), CV_32FC1);

    // timing
    float elapsedTime = 0.0;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    RDLWavelet rdlw(rows, cols, 4);

    // start event recording
    cudaEventRecord(startEvent);
    // do RDL Wavelet
    rdlw.doRDLWavelet(imgOuts, imgIn);
    rdlw.doInverseRDLWavelet(imgRestore);
    // stop event recording
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    cout << "Total timing: " << elapsedTime << " ms." << endl;

    // destroy events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // show result
    cv::imshow("Restored Image", imgRestore);
    cv::waitKey();

    return 0;
}
