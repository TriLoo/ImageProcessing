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
    // however, std::vector is not thread safe
    // nproc = 4
    vector<Mat> layersA(0), layersB(0);
    thread *ptRDL_A, *ptRDL_B, *ptWM;
    //rdlPimpl->RdlWavelet(layersA, imgInA);
    ptRDL_A = new thread(&RDLWavelet::RdlWavelet, rdlPimpl, std::ref(layersA), std::ref(imgInA));
    //rdlPimpl->RdlWavelet(layersB, imgInB);
    ptRDL_B = new thread(&RDLWavelet::RdlWavelet, rdlPimpl, std::ref(layersB), std::ref(imgInB));
    endPoints.push_back(chrono::steady_clock::now());

    //cout << "Success 2." << endl;
    // calculate weighted map
    // calculation saliency map
    vector<Mat> wmBase(0), wmDetail(0);
    vector<Mat> imgIns = {imgInA, imgInB};
    //cout << imgIns.size() << endl;
    wmPimpl->setParams();
    //wmPimpl->weightedmap(wmBase, wmDetail, imgIns);
    ptWM = new thread(&WeightedMap::weightedmap, wmPimpl, std::ref(wmBase), std::ref(wmDetail), std::ref(imgIns));

    // threads synchronize
    ptWM->join();
    ptRDL_A->join();
    ptRDL_B->join();
    endPoints.push_back(chrono::steady_clock::now());


    assert(layersA.size() == 4);
    assert(layersB.size() == 4);
    assert(wmBase.size() == 2);
    assert(wmDetail.size() == 2);


    // Fusion the coefficients
    // Base layers fusion
    // mul: element-wise multiplication
    layersA[0] = (layersA[0].mul(0.5 + 0.5 * (wmBase[0] - wmBase[1]))) + layersB[0].mul(0.5 + 0.5 * (wmBase[1] - wmBase[0]));

    // detail layers fusion
    for (int i = 1; i < 4; ++i)
        layersA[i] = layersA[i].mul(wmDetail[0]) + layersB[i].mul(wmDetail[1]);
    endPoints.push_back(chrono::steady_clock::now());


    // inverse RDL Wavelet transform
    rdlPimpl->inverseRdlWavelet(imgOut, layersA);

    endPoints.push_back(chrono::steady_clock::now());

    chrono::duration<double> elapsedTime;
    chrono::steady_clock::time_point freEndPoint = startPoints;

    int Step = 0;

    for (auto & ele : endPoints)
    {
        elapsedTime = chrono::duration_cast<chrono::duration<double> >(ele - freEndPoint);
        // Step 1: RDLWavelet decomposition of two input images
        // Step 2: Calculation of Weighted Map
        // Step 3: Calculation of final fused two layers
        // Step 4: inverse RDLWavelet synthesis
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
