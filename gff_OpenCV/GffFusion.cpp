//
// Created by smher on 17-10-10.
//

#include "GffFusion.h"

#define GUIRAD_B 45
#define GUIEPS_B 0.3
#define GUIRAD_D 10
#define GUIEPS_D 0.0001

void GffFusion::TwoScale(const Mat &inA, Size ksize, Mat &outH, Mat &outL)
{
    // use average filter to get the twoscale decomposing
    //clock_t start, stop;
    //start = clock();
    blur(inA, outL, ksize);
    //stop = clock();
    //cout << "Boxfilter on Opencv : " << 1000.0 * (stop - start) / CLOCKS_PER_SEC << " ms" << endl;
    outH = inA - outL;
}

void GffFusion::SaliencyMap(Mat &imgIn, Mat &SMap)
{
    // Do the Laplace filters 3 * 3
    Laplacian(imgIn, SMap, CV_32F, 3);
    // get the absolute values of 'SMap'
    SMap = abs(SMap);

    // Apply Gaussian filter on SMap
    GaussianBlur(SMap, SMap, Size(5, 5), Sigma);
}

void GffFusion::WeightMap(Mat &imgInA, Mat &imgInB, vector<Mat *> &vecA, vector<Mat *> &vecB, vector<Mat *> &vec)
{
    // get the number of input images to be fused
    int noImg = vec.size();
    int row = imgInA.rows;
    int col = imgInA.cols;

    assert(noImg == 2);

    // declare two temp Mat
    Mat imgMapA = Mat::zeros(row, col, CV_32F);
    Mat imgMapB = Mat::zeros(row, col, CV_32F);

    // get the P value by compare each mat
    // vec[0] : imgInA; vec[1] : imgInB.
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
        {
            if(vec[0]->at<float>(i, j) > vec[1]->at<float>(i, j))
                imgMapA.at<float>(i, j) = 1;
            else
                imgMapB.at<float>(i, j) = 1;
        }

    // use guided filter to get the weighted map, ----  cannot find this function
    clock_t start, stop;
    start = clock();
    //*(vecA[0]) = guidedFilter(imgInA, imgMapA, 20, 0.1, CV_32F);
    *(vecA[0]) = guidedFilter(imgInA, imgMapA, GUIRAD_B, GUIEPS_B, CV_32F);
    stop = clock();
    cout << "Guided Filter Time : " << (stop - start) * 1.0 / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
    *(vecA[1]) = guidedFilter(imgInA, imgMapA, GUIRAD_D, GUIEPS_D, CV_32F);

    *(vecB[0]) = guidedFilter(imgInB, imgMapB, GUIRAD_B, GUIEPS_B, CV_32F);
    *(vecB[1]) = guidedFilter(imgInB, imgMapB, GUIRAD_D, GUIEPS_D, CV_32F);

    // test for bilateralFilter ----   can find this function
    //bilateralFilter(imgInA, *(vec[0]), 10, 1, 1);
}

void GffFusion::TwoScaleRec(vector<Mat *> &vecW, vector<Mat *> &LayerIn, Mat &LayerOut)
{
    LayerOut = vecW[0]->mul(*LayerIn[0]) + vecW[1]->mul(*LayerIn[1]);
}

void GffFusion::gffFusion(Mat &imgA, Mat &imgB, Mat &Res)
{
    int row = imgA.rows;
    int col = imgA.cols;

    cout << "The dimension of Image is : " << row << " * " << col << endl;

    // declare two more Mats to store the two scale result
    Mat outA_B = Mat::zeros(row, col, CV_32F);
    Mat outA_D = Mat::zeros(row, col, CV_32F);

    Mat outB_B = Mat::zeros(row, col, CV_32F);
    Mat outB_D = Mat::zeros(row, col, CV_32F);

    // convert the input image to float
    imgA.convertTo(imgA, CV_32F, 1.0);
    imgB.convertTo(imgB, CV_32F, 1.0);


    // Decomposing the image into tow scale parts
    TwoScale(imgA, KSIZE, outA_D, outA_B);   // outA: the high pass, outB: the low pass
    TwoScale(imgB, KSIZE, outB_D, outB_B);   // outA: the high pass, outB: the low pass
    /*
	// test the decomposing results
	outA_B.convertTo(outA_B, CV_8UC1, 1.0);
	outB_B.convertTo(outB_B, CV_8UC1, 1.0);
	imshow("output A", outA_B);
	imshow("output B", outB_B);
	waitKey(0);

	outA_B.convertTo(outA_B, CV_32F, 1.0);
	outB_B.convertTo(outB_B, CV_32F, 1.0);
    */

    // get the  Saliency Mpas of input images
    Mat SalMapA = Mat::zeros(row, col, CV_32F);
    Mat SalMapB = Mat::zeros(row, col, CV_32F);
    SaliencyMap(imgA, SalMapA);
    SaliencyMap(imgB, SalMapB);

    // Get the Weighted Maps based on Saliency Map and Guided Filter
    //void WeightMap(Mat &imgInA, Mat &imgInB, vector<Mat *> &vecA, vector<Mat *> &vecB, vector<Mat *> &vec)
    Mat WeiMapA_B = Mat::zeros(row, col, CV_32F);
    Mat WeiMapA_D = Mat::zeros(row, col, CV_32F);
    Mat WeiMapB_B = Mat::zeros(row, col, CV_32F);
    Mat WeiMapB_D = Mat::zeros(row, col, CV_32F);

    // Test for the Vector<Mat *>
    vector<Mat *> tempPtrA = {&WeiMapA_B, &WeiMapA_D};
    vector<Mat *> tempPtrB = {&WeiMapB_B, &WeiMapB_D};
    vector<Mat *> tempPtrMap = {&SalMapA, &SalMapB};

    WeightMap(imgA, imgB, tempPtrA, tempPtrB, tempPtrMap);
    //WeightMap(imgA, imgB, vector<Mat *>({&WeiMapA_B, &WeiMapA_D}), vector<Mat *>({&WeiMapB_B, &WeiMapB_D}), vector<Mat *>({&SalMapA, &SalMapB}));

    // Get the restored fusion result by combine the base and detail layers
    //void TwoScaleRec(vector<Mat *> &vecW, vector<Mat *> &LayerIn, Mat &LayerOut)
    tempPtrA.clear();
    tempPtrB.clear();
    tempPtrA = {&WeiMapA_B, &WeiMapB_B};
    tempPtrB = {&outA_B, &outB_B};
    Mat LayerBase = Mat::zeros(row, col, CV_32F);
    Mat LayerDetail = Mat::zeros(row, col, CV_32F);

    TwoScaleRec(tempPtrA, tempPtrB, LayerBase);
    //TwoScaleRec(vector<Mat *>({&WeiMapA_B, &WeiMapB_B}), vector<Mat *>({&outA_B, &outB_B}), LayerBase);
    tempPtrA = {&WeiMapA_D, &WeiMapB_D};
    tempPtrB = {&outA_D, &outB_D};
    TwoScaleRec(tempPtrA, tempPtrB, LayerDetail);
    //TwoScaleRec({&WeiMapA_D, &WeiMapB_D}, tempPtrB, LayerDetail);    // Cannot work!
    //TwoScaleRec(vector<Mat *>({&WeiMapA_D, &WeiMapB_D}), vector<Mat *>({&outA_D, &outB_D}), LayerDetail);  // Cannot work!

    // Add the base layer and detail layer to obtain the finnal result
    //Mat Result = Mat::zeros(row, col, CV_32F);
    Res = LayerBase + LayerDetail;
}

void GffFusion::gffFusionColor(Mat &imgA, Mat &imgB, Mat &Res)
{
    try
    {
        if(imgA.rows != imgB.rows || imgA.cols != imgB.cols || imgA.channels() != imgB.channels())
            throw runtime_error("Dimension doesn't match!");
    }
    catch(runtime_error err)
    {
        cerr << err.what() << endl;
        cout << "Try Again (y/n): " << endl;
        char ans = 'y';
        cin >> ans;
        if(ans == 'Y' || ans == 'y')
        {
            cout << "Input tow images' name " << endl;
            vector<string> names;
            string t;
            cin >> t;
            names.push_back(t);
            cin >> t;
            names.push_back(t);
            imread(names[0], IMREAD_COLOR);
            imread(names[1], IMREAD_COLOR);
        }
        else
            exit(EXIT_FAILURE);
    }
    int row = imgA.rows, col = imgA.cols;
    int channs = imgA.channels();
    cout << "Channels = " << channs << endl;

    // separate color channels
    Mat bgrA[3], bgrB[3], outF[3];
    split(imgA, bgrA);
    split(imgB, bgrB);

    cout << "Depth of Image A: " << imgA.depth() << endl;

    // test the results, save the results based on boost!
    boost::format fmt("%s_%d.%s");   // color_i.jpg
    //cout << "imgA.channels = " << imgA.channels() << endl;
    if(channs == 3)
    {
        cout << "Begin boost::thread..." << endl;
        //boost::thread t0(&GffFusion::gffFusion, this, bgrA[0], bgrB[0], outF[0]);
        //boost::thread t0([&, bgrA, bgrB, outF]{this->gffFusion(bgrA[0], bgrB[0], outF[0]);});
        boost::thread t0(bind(&GffFusion::gffFusion, this, bgrA[0], bgrB[0], outF[0]));
        boost::thread t1(bind(&GffFusion::gffFusion, this, bgrA[1], bgrB[1], outF[1]));
        boost::thread t2(bind(&GffFusion::gffFusion, this, bgrA[2], bgrB[2], outF[2]));
        t0.join();
        t1.join();
        t2.join();
        cout << "blue width = " << outF[0].size() << endl;
        //cout << "blue width = " << bgrA[0].size() << endl;
    }
    /*
    for(int i = 0; i < channs; i++)
    {
        //imwrite((fmt%"color"%i%"jpg").str(), bgrA[i]);
        gffFusion(bgrA[i], bgrB[i], outF[i]);
    }
    */

    merge(outF, channs, Res);
}


