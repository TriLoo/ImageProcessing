//
// Created by smher on 18-1-11.
//

#include "Fusion.h"

using namespace std;
using namespace cv;

void imgShow(const Mat& img)
{
    imshow("Temp", img);
    waitKey(0);
}

void imgShow(const Mat&& img)
{
    imshow("Temp", img);
    waitKey(0);
}

Fusion::Fusion(int r, int c) : rdlPimpl(new RDLWavelet(r, c)), wmPimpl(new WeightedMap(r, c))
{
}

Fusion::~Fusion()
{
}

void Fusion::imageFusion(cv::Mat &imgOut, const cv::Mat &imgInA, const cv::Mat &imgInB)
{
    assert(imgInA.channels() == 1);
    assert(imgInB.channels() == 1);

    //cout << "Success 1." << endl;

    // decomposing input image input base layers and detail layers
    vector<Mat> layersA(0), layersB(0);
    rdlPimpl->RdlWavelet(layersA, imgInA);
    rdlPimpl->RdlWavelet(layersB, imgInB);

    //imgShow(layersB[0]);

    //cout << "Success 2." << endl;

    // calculation saliency map
    vector<Mat> wmBase(0), wmDetail(0);
    vector<Mat> imgIns = {imgInA, imgInB};
    //cout << imgIns.size() << endl;
    wmPimpl->setParams();
    wmPimpl->weightedmap(wmBase, wmDetail, imgIns);

    //imgShow(wmDetail[0]);

    // Fusion the coefficients
    assert(layersA.size() == 4);
    assert(layersB.size() == 4);
    assert(wmBase.size() == 2);
    assert(wmDetail.size() == 2);
    // Base layers fusion
    layersA[0] = layersA[0].mul(wmBase[0]) + layersB[0].mul(wmBase[1]);

    //imgShow(layersA[0]);

    // detail layers fusion
    for (int i = 1; i < 4; ++i)
        layersA[i] = layersA[i].mul(wmDetail[0]) + layersB[i].mul(wmDetail[1]);
    //imgShow(layersA[1]);

    // inverse RDL Wavelet transform
    //Mat tempMat;
    //rdlPimpl->inverseRdlWavelet(tempMat, layersA);
    rdlPimpl->inverseRdlWavelet(imgOut, layersA);
    //imgShow(imgOut);

    //imgOut = tempMat;
}

