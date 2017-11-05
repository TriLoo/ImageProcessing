#include "ImageMosaic.h"
#include "chrono"

// The left image is the 'dst' image
// The right image is the 'src' image
int main()
{
    Mat img1 = imread("1.jpg", IMREAD_COLOR);
    Mat img2 = imread("2.jpg", IMREAD_COLOR);
    //Mat img1 = imread("left.jpg", IMREAD_COLOR);
    //Mat img2 = imread("right.jpg", IMREAD_COLOR);


    if(!img1.data || !img2.data)
    {
        cerr << "Read Image Failed ..." << endl;
        return -1;
    }

    cout << "Input Size : " << img1.size() << endl;

    vector<Mat *> inImgs = {&img1, &img2};

    Mat outImg1, outImg2;
    vector<Mat *> outImgs = {&outImg1, &outImg2};

    ImageMosaic im;

    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    //im.FeatureMatchTest(outImgs, inImgs);
    im.imageMosaicTest(&outImg1, inImgs);
    //im.imageRegisterTest(&outImg1, inImgs);

    chrono::steady_clock::time_point stop = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(stop - start);
    cout << "Time used : " << time_used.count() << " s." << endl;

    //imshow("Origin Match", *outImgs[0]);
    //imshow("Good Match", *outImgs[1]);

    //waitKey(0);

    return 0;
}

/*
int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat img = imread("lena.jpg", IMREAD_COLOR);

    if(!img.data)
    {
        cerr << "Read Image failed ..." << endl;
        return 1;
    }


    cout << "Input Size: " << img.size() << endl;
    imshow("Input", img);
    Mat Out = Mat::zeros(img.size(), img.depth());

    ImageMosaic im;

    im.FeatureDetectorTest(&Out, &img);
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    im.FeatureDetectorTest(&Out, &img);
    chrono::steady_clock::time_point stop = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(stop - start);
    cout << "Time used : " << time_used.count() << " s." << endl;

    imshow("Output", Out);

    waitKey(0);

    return 0;
}
*/
