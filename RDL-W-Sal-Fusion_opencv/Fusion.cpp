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

    chrono::steady_clock::time_point startPoints;
    vector<chrono::steady_clock::time_point> endPoints;

    //cout << "Success 1." << endl;

    startPoints = chrono::steady_clock::now();
    // decomposing input image input base layers and detail layers
    vector<Mat> layersA(0), layersB(0);
    rdlPimpl->RdlWavelet(layersA, imgInA);
    // the decomposition is not correct
    //Mat temp;
    //normalize(layersA[0], temp, 0.0, 1.0, NORM_MINMAX);
    //temp.convertTo(temp, CV_8UC1, 255);
    //imwrite("temp.png", temp);
    //imshow("CA", layersA[0]);
    //imshow("CH", layersA[1]);
    //imshow("CV", layersA[2]);
    //imshow("CD", layersA[3]);
    //waitKey(0);
    rdlPimpl->RdlWavelet(layersB, imgInB);
    //imshow("CA", layersB[0]);
    //imshow("CH", layersB[1]);
    //imshow("CV", layersB[2]);
    //imshow("CD", layersB[3]);
    //waitKey(0);
    endPoints.push_back(chrono::steady_clock::now());

    //imgShow(layersB[0]);

    //cout << "Success 2." << endl;

    // calculation saliency map
    vector<Mat> wmBase(0), wmDetail(0);
    vector<Mat> imgIns = {imgInA, imgInB};
    //cout << imgIns.size() << endl;
    wmPimpl->setParams();
    wmPimpl->weightedmap(wmBase, wmDetail, imgIns);

    endPoints.push_back(chrono::steady_clock::now());

    //imgShow(wmDetail[0]);

    // Fusion the coefficients
    assert(layersA.size() == 4);
    assert(layersB.size() == 4);
    assert(wmBase.size() == 2);
    assert(wmDetail.size() == 2);
    // Base layers fusion
    //layersA[0] = layersA[0].mul(wmBase[0]) + layersB[0].mul(wmBase[1]);
    // mul: element-wise multiplication
    layersA[0] = (layersA[0].mul(0.5 + 0.5 * (wmBase[0] - wmBase[1]))) + layersB[0].mul(0.5 + 0.5 * (wmBase[1] - wmBase[0]));

    //imgShow(layersA[0]);

    // detail layers fusion
    for (int i = 1; i < 4; ++i)
        layersA[i] = layersA[i].mul(wmDetail[0]) + layersB[i].mul(wmDetail[1]);
    //imgShow(layersA[1]);

    endPoints.push_back(chrono::steady_clock::now());

    // inverse RDL Wavelet transform
    //Mat tempMat;
    //rdlPimpl->inverseRdlWavelet(tempMat, layersA);
    rdlPimpl->inverseRdlWavelet(imgOut, layersA);
    //imgShow(imgOut);

    endPoints.push_back(chrono::steady_clock::now());

    chrono::duration<double> elapsedTime;
    chrono::steady_clock::time_point freEndPoint = startPoints;

    int Step = 0;

    for (auto & ele : endPoints)
    {
        elapsedTime = chrono::duration_cast<chrono::duration<double> >(ele - freEndPoint);
        cout << "Step " << Step++ << ": using " << elapsedTime.count() * 1000.0 << " ms.startPoints" << endl;
        freEndPoint = ele;
    }
}

void Fusion::imageFusionColor(cv::Mat &imgOut, const cv::Mat &imgInA, const cv::Mat &imgInB)
{
    assert(imgInA.channels() == 3);
    assert(imgInB.channels() == 3);

    vector<Mat> imgInA_RGB, imgInB_RGB, imgOut_RGB;
    split(imgInA, imgInA_RGB);
    split(imgInB, imgInB_RGB);

    Mat inA, inB, Out;
    for(int i = 0; i < 3; ++i)
    {
        inA = imgInA_RGB[i];
        inB = imgInB_RGB[i];
        imageFusion(Out, inA, inB);
        imgOut_RGB.push_back(Out);
    }

    merge(imgOut_RGB, imgOut);
}
