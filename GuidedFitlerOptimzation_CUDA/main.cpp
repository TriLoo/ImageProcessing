#include <iostream>
#include "GuidedFilter.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    //cout << "sizeof(float4): " << sizeof(float4) << endl;        // return 16 bytes

    //Mat imgI = imread("Barbara.jpg", IMREAD_COLOR);                // no padding
    //Mat imgP = imread("Barbara.jpg", IMREAD_COLOR);
    //Mat imgI = imread("lena.jpg", IMREAD_COLOR);                 // padded from 396 to 416
    //Mat imgP = imread("lena.jpg", IMREAD_COLOR);
    //Mat imgI = imread("img_00000.bmp", IMREAD_COLOR);            // no padding
    //Mat imgP = imread("img_00000.bmp", IMREAD_COLOR);
    Mat imgI = imread("tulips.png", IMREAD_COLOR);                // no padding
    Mat imgP = imread("tulips.png", IMREAD_COLOR);

    assert(imgI.empty() == false);
    assert(imgP.empty() == false);

    cout << "Image Infor:"  << endl <<
         "Rows: " << imgI.rows << endl <<
         "Cols: " << imgI.cols << endl <<
         "Channels: " << imgI.channels() << endl;

    imshow("Input I", imgI);
    imshow("Input P", imgP);

    Mat imgOut = Mat::zeros(imgI.size(), CV_32FC4);    // size(): return x - cols, y - rows
    //Mat imgOut = Mat::zeros(imgI.size(), CV_32FC3);    // size(): return x - cols, y - rows

    GFilter gf(imgI.rows, imgI.cols);

    //imgI.convertTo(imgI, CV_32FC3, 1.0/255);
    //imgP.convertTo(imgP, CV_32FC3, 1.0/255);
    //gf.guidedfilterOpenCV(imgOut, imgI, imgP);
    //gf.guidedfilter(imgOut, imgI, imgP);

    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    // TODO: when the eps is less than zero, the result is wrong ! ! !
    // I guess, that may resulted by the positions of variables in some func. are wrong !
    // Solved! Because the calculation of corrI & corrIp, 在纵向计算时，并不需要进行平方操作，而尽在横向进行一次乘法操作即可
    gf.guidedfilter(imgOut, imgI, imgP);
    //gf.boxfilterTest(imgOut, imgI);                         // lena,   // update: correct !


    //gf.guidedfilterOpenCV(imgOut, imgI, imgP);           // lena: (no cache) 77.2914 ms, (with cache): 15.12ms
    //gf.boxfilterNpp(imgOut, imgI, 45);                 // return lena: 12.858 ms
    //cudaDeviceSynchronize();
    chrono::steady_clock::time_point stop = chrono::steady_clock::now();
    chrono::duration<double> elapsedTime = chrono::duration_cast<chrono::duration<double>>(stop - start);
    cout << "Guided Filter on GPU: " << elapsedTime.count() * 1000.0  << " ms." << endl;
    //cout << "Guided Filter on CPU: " << elapsedTime.count() * 1000.0  << " ms." << endl;
    /*
    cvtColor(imgI, imgI, CV_BGR2BGRA);
    cout << "Channels = " << imgI.channels() << endl;


    if (imgI.type() == CV_8UC4)
        cout << "The type is: " << CV_8UC4 << endl;

    if (imgI.type() != CV_32FC4)
    {
        cout << "The type is: " << imgI.type() << endl;
        cout << "The CV_32FC4 is: " << CV_32FC4 << endl;

        imgI.convertTo(imgOut, CV_32FC4, 1.0/255);
        cout << "Now type is: " << imgI.type() << endl;

        cout << "Channels = " << imgI.channels() << endl;
    }
    */

    cvtColor(imgOut, imgOut, CV_BGRA2BGR);
    //normalize(imgOut, imgOut, 0, 1.0, NORM_MINMAX);

    imshow("Result", imgOut);

    waitKey(0);

    return 0;
}
