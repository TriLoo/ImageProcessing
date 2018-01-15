#include "WeightedMap.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    // test weighted map class
    Mat imgA = imread("source20_1.tif", IMREAD_GRAYSCALE);
    Mat imgB = imread("source20_2.tif", IMREAD_GRAYSCALE);
    assert(imgA.empty() != true);
    assert(imgB.empty() != true);

    const int row = imgA.rows;
    const int col = imgA.cols;

    // prepare parameters
    //double c = 0.95;
    WeightedMap wm(row, col);
    wm.setParams();

    Mat res;
    vector<Mat> baseWM(0), detailWM(0);
    vector<Mat> imgIns = {imgA, imgB};
    wm.weightedmap(baseWM, detailWM, imgIns);

    assert(baseWM.size() == 2);
    Mat tempMat = baseWM[0];
    normalize(tempMat, tempMat, 0, 255, NORM_MINMAX);

    imshow("Output", tempMat);

    waitKey(0);

    /*
    // For test vector.clear()
    vector<int> a = {1, 2, 3, 4};
    cout << "Before: " << a.size() << endl;
    a.clear();
    cout << "After: " << a.size() << endl;
    a.push_back(1);
    cout << "Final: " << a.size() << endl;
    */

    return 0;
}