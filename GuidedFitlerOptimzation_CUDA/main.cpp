#include "GuidedFilter.h"

using namespace std;
using namespace cv;

int main() {
    //std::cout << "Hello, World!" << std::endl;
    Mat imgI = imread("Barbara.jpg", IMREAD_COLOR);
    //assert(imgI.empty() != true);
    assert(!imgI.empty());
    cout << "Image Info: " << endl;
    cout << "Size: " << imgI.rows << " * " << imgI.cols << endl;
    cout << "Channels: " << imgI.channels() << endl << endl;

    imgI.convertTo(imgI, CV_32FC3, 1.0 / 255);
    imshow("Input", imgI);
    Mat imgP = imgI;

    GFilter gf(imgI.rows, imgI.cols);
    gf.setParams(16, 0.01);

    Mat imgOut;
    //gf.guidedfilterOpenCV(imgOut, imgI, imgP);
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    gf.guidedfilterOpenCV(imgOut, imgI, imgP);
    chrono::steady_clock::time_point stop = chrono::steady_clock::now();
    chrono::duration<double> elapsedTime = static_cast<chrono::duration<double>>(stop - start);
    cout << "Used time: " << elapsedTime.count() * 1000.0 << "ms." << endl;

    //normalize(imgOut, imgOut, 0, 1, CV_MINMAX);
    imshow("Result", imgOut);
    waitKey(0);

    return 0;
}
