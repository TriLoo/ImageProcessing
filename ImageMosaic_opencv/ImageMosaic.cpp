//
// Created by smher on 17-11-4.
//

#include "ImageMosaic.h"

/*
ImageMosaic::ImageMosaic(const Mat &H, const Mat &src)
{
    corners_ = CalcCorners(H, src);
}
*/

ImageMosaic::~ImageMosaic()
{
}

// H is the transform matrix
four_corners_t ImageMosaic::CalcCorners(const Mat &H, const Mat &src)
{
    four_corners_t corners;
    double v2[] = {0, 0, 1};   // left top corner
    double v1[3];              //

    Mat V2 = Mat(3, 1, CV_64FC1, v2);
    Mat V1 = Mat(3, 1, CV_64FC1, v1);    // V1 is the 'reference' of v1, see OpenCV Doc.
    V1 = H * V2;
    cout << "V1 = " << V1 << endl;
    // Left-Top Corner : (0, 0, 1)
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    // Left-Bottom Corner : (0, src.rows, 1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);
    V1 = Mat(3, 1, CV_64FC1, v1);
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    // Right-Top Corner : (src.cols, 0, 1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);
    V1 = Mat(3, 1, CV_64FC1, v1);
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    // Right-Bottom Corner : (src.cols, src.rows, 1)
    //v2 = {src.cols, src.rows, 1};
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);
    V1 = Mat(3, 1, CV_64FC1, v1);
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];

    return corners;
}

void ImageMosaic::OptimizeSeam(Mat &img1, Mat &trans, Mat &dst)
{
    int start = MIN(corners_.left_top.x, corners_.left_bottom.x);

    double processWidth = img1.cols - start;
    int rows = dst.rows;
    int cols = img1.cols;

    double alpha = 1;

    for(int i = 0; i < rows; i++)
    {
        uchar *p = img1.ptr<uchar>(i);        // the i-th line first address
        uchar *t = trans.ptr<uchar>(i);
        uchar *d = dst.ptr<uchar>(i);

        for(int j = start; j < cols; j++)
        {
            if(t[j*3] == 0 && t[j*3+1] == 0 && t[j * 3 + 2] == 0)
            {
                alpha = 1;
            }
            else
            {
                // 权重，与当前像素点距离重叠区域左边界的距离成正比
                alpha = (processWidth - (j - start)) / processWidth;
            }

            d[j * 3] = p[j * 3]  * alpha + t[j * 3] * (1 - alpha);
            d[j * 3 + 1] = p[j * 3 + 1]  * alpha + t[j * 3 + 2] * (1 - alpha);
            d[j * 3 + 2] = p[j * 3 + 1]  * alpha + t[j * 3 + 2] * (1 - alpha);
        }
    }
}

void ImageMosaic::FeatureDetector(Mat *out, Mat *in)
{
    vector<KeyPoint> keypoints;

    Mat descriptors;

    //Ptr<ORB> orb = ORB::create(500, 1.2f, )
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    // Step 1 : detect Oriented FAST
    orb->detect(*in, keypoints);

    // Step 2 : calculate Brief descriptors
    orb->compute(*in, keypoints, descriptors);

    // Step 3 : Show the result
    // Mat output1;
     drawKeypoints(*in, keypoints, *out, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // imshow("ORB Features", *out);

    //waitKey(0);
}

void ImageMosaic::FeatureDetectorTest(Mat *out, Mat *in)
{
    FeatureDetector(out, in);
}

void ImageMosaic::FeatureMatch(vector<Mat *> out, vector<Mat *> in)
{
}

void ImageMosaic::FeatureMatchTest(vector<Mat *> &out, vector<Mat *> &in)
{
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    // Step 1 : feature detect
    orb->detect(*in[0], keypoints_1);
    orb->detect(*in[1], keypoints_2);

    // Step 2 : descriptor calculation
    orb->compute(*in[0], keypoints_1, descriptors_1);
    orb->compute(*in[1], keypoints_2, descriptors_2);
    //cout << "Setp 2 : Success." << endl;

    // Step 3 : match the descriptors
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);
    //cout << "Setp 3 : Success." << endl;

    // Step 4 : get the good matches
    double min_dist = 10000, max_dist = 0;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }

    //cout << "Minmum distance : " << min_dist << endl;
    //cout << "Maxmum distance : " << max_dist << endl;

    vector<DMatch> good_matches;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        if(matches[i].distance < max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    cout << "Origin Matches Size : " << matches.size() << endl;
    cout << "Good Matches Size : " << good_matches.size() << endl;

    //cout << "Setp 4 : Success." << endl;

    // Step 5 : draw the result
    //Mat img_match;
    //Mat img_goodmatch;
    drawMatches(*in[0], keypoints_1, *in[1], keypoints_2, matches, *out[0]);
    drawMatches(*in[0], keypoints_1, *in[1], keypoints_2, good_matches, *out[1]);

    //imshow("After All Match", img_match);
    //imshow("Good Match", img_goodmatch);

    //waitKey(0);
}

void ImageMosaic::imageMosaic(Mat *outs, vector<Mat *> &ins)
{

}

void ImageMosaic::imageMosaicTest(Mat *outs, vector<Mat *> ins)
{
    assert(ins.size() >= 2);
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    // Step 1 : feature detect
    orb->detect(*ins[0], keypoints_1);
    orb->detect(*ins[1], keypoints_2);

    // Step 2 : descriptor calculation
    orb->compute(*ins[0], keypoints_1, descriptors_1);
    orb->compute(*ins[1], keypoints_2, descriptors_2);

    // Step 3 : match the descriptors
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);

    // Step 4 : get the good matches
    double min_dist = 10000, max_dist = 0;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }

    //cout << "Minmum distance : " << min_dist << endl;
    //cout << "Maxmum distance : " << max_dist << endl;

    vector<DMatch> good_matches;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        //if(matches[i].distance < max(3 * min_dist, 30.0))
        if(matches[i].distance < 3 * min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }

    cout << "Origin Matches Size : " << matches.size() << endl;
    cout << "Good Matches Size : " << good_matches.size() << endl;

    // Step 5 : image Register
    vector<Point2f> imagePoints1, imagePoints2;

    for(int i = 0; i < good_matches.size(); i++)
    {
        //imagePoints1.push_back(keypoints_1[good_matches[i].trainIdx].pt);
        //imagePoints2.push_back(keypoints_2[good_matches[i].queryIdx].pt);
        imagePoints1.push_back(keypoints_1[good_matches[i].queryIdx].pt);         // queryIdx refers to keypoints1 (dst image)
        imagePoints2.push_back(keypoints_2[good_matches[i].trainIdx].pt);         // trainIdx refers to keypoints2 (src image)
    }

    // get the 3 * 3 matrix
    cout << "Before findHomography" << endl;
    Mat homo = findHomography(imagePoints2, imagePoints1, RANSAC);                // src, dst, method
    //Mat homo = findHomography(imagePoints1, imagePoints2);
    cout << "Transform Matrix : " << homo << endl;

    //Mat imageTransform1, imageTransform2;
    Mat imageTransform1;
    corners_ = CalcCorners(homo, *ins[1]);
    // Apply a perspective transformation to the first image
    // for test
    cout << "Corner.left_top : " << corners_.left_top << endl;
    cout << "Corner.left_bottom : " << corners_.left_bottom << endl;
    cout << "Corner.right_top: " << corners_.right_top << endl;
    cout << "Corner.right_bottom : " << corners_.right_bottom << endl;
    //Mat shftMat = (Mat_<double>(3, 3) << 1.0, 0, (*ins[0]).cols, 0, 1.0, 0, 0, 0, 1.0);      // used for the left image is src image
    warpPerspective(*ins[1], imageTransform1, homo, Size(MAX(corners_.right_top.x, corners_.right_bottom.x), (*ins[0]).rows));
    //warpPerspective(*ins[0], imageTransform1, homo, Size(MAX(corners_.right_top.x, corners_.right_bottom.x), (*ins[1]).rows));
    //warpPerspective(*ins[0], imageTransform1, shftMat * homo, Size(MAX(corners_.right_top.x, corners_.right_bottom.x), (*ins[1]).rows));

    // For test
    //cout << "After Transformed : " << imageTransform1.size() << endl;
    imshow("Transformed Image", imageTransform1);
    //imshow("Transformed Image", *ins[1]);
    waitKey(0);

    // Step 6 : Image Copy
    int dst_width = imageTransform1.cols;
    int dst_height = (*ins[0]).rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    cout << "Input size : " << (*ins[1]).size() << endl;
    cout << "Result Size : " << dst.size() << endl;
    cout << "Result Size : " << dst.rows << endl;

    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    cout << "First copy Success." << endl;
    (*ins[0]).copyTo(dst(Rect(0, 0, (*ins[0]).cols, (*ins[0]).rows)));
    //(*ins[1]).copyTo(dst(Rect(imageTransform1.cols, 0, (*ins[1]).cols, (*ins[1]).rows)));
    cout << "Second copy Success." << endl;
    // Step 7 : Fusion the overlapping area based on weighted averate
    OptimizeSeam(*ins[0], imageTransform1, dst);
    imshow("b_dst", dst);

    // Step 5 : draw the result
    //Mat img_match;
    //Mat img_goodmatch;
    //drawMatches(*ins[0], keypoints_1, *ins[1], keypoints_2, matches, outs);
    drawMatches(*ins[0], keypoints_1, *ins[1], keypoints_2, good_matches, *outs);

    //imshow("After All Match", img_match);
    //imshow("Good Match", img_goodmatch);

    waitKey(0);
}

void ImageMosaic::imageRegister(Mat *outs, vector<Mat *> &ins)
{
}

void ImageMosaic::imageRegisterTest(Mat *outs, vector<Mat *> &ins)
{
    assert(ins.size() >= 2);
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    // Step 1 : feature detect
    orb->detect(*ins[0], keypoints_1);
    orb->detect(*ins[1], keypoints_2);

    // Step 2 : descriptor calculation
    orb->compute(*ins[0], keypoints_1, descriptors_1);
    orb->compute(*ins[1], keypoints_2, descriptors_2);

    // Step 3 : match the descriptors
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);

    // Step 4 : get the good matches
    double min_dist = 10000, max_dist = 0;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }

    //cout << "Minmum distance : " << min_dist << endl;
    //cout << "Maxmum distance : " << max_dist << endl;

    vector<DMatch> good_matches;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        if(matches[i].distance < max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    cout << "Origin Matches Size : " << matches.size() << endl;
    cout << "Good Matches Size : " << good_matches.size() << endl;

    // Step 5 : image Register
    vector<Point2f> imagePoints1, imagePoints2;

    for(int i = 0; i < good_matches.size(); i++)
    {
        imagePoints1.push_back(keypoints_1[good_matches[i].trainIdx].pt);
        imagePoints2.push_back(keypoints_2[good_matches[i].queryIdx].pt);
    }

    // get the 3 * 3 matrix
    cout << "Before findHomography" << endl;
    Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
    cout << "Transform Matrix : " << homo << endl;

    //Mat imageTransform1, imageTransform2;
    Mat imageTransform1;
    corners_ = CalcCorners(homo, *ins[0]);
    // Apply a perspective transformation to the first image
    // for test
    cout << "Corner.left_top : " << corners_.left_top << endl;
    cout << "Corner.left_bottom : " << corners_.left_bottom << endl;
    cout << "Corner.right_top: " << corners_.right_top << endl;
    cout << "Corner.right_bottom : " << corners_.right_bottom << endl;
    warpPerspective(*ins[0], imageTransform1, homo, Size(MAX(corners_.right_top.x, corners_.right_bottom.x), (*ins[1]).rows));

    // For test
    //cout << "After Transformed : " << imageTransform1.size() << endl;
    //imshow("Transformed Image", imageTransform1);
    //imshow("Transformed Image", *ins[1]);
    //waitKey(0);

    // Step 6 : Image Copy
    int dst_width = imageTransform1.cols + (*ins[1]).cols;
    int dst_height = (*ins[1]).rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    /*
    cout << "Input size : " << (*ins[1]).size() << endl;       // 766 * 427
    cout << "Result Size : " << dst.size() << endl;            // 553 * 427
    cout << "Result Size : " << dst.rows << endl;              // 427

    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    cout << "First copy Success." << endl;
    (*ins[1]).copyTo(dst(Rect(imageTransform1.cols, 0, (*ins[1]).cols, (*ins[1]).rows)));
    cout << "Second copy Success." << endl;

    imshow("b_dst", dst);
    */

    // Step 7 : Fusion the overlapping area based on weighted averate


    // Step 5 : draw the result
    //Mat img_match;
    //Mat img_goodmatch;
    //drawMatches(*ins[0], keypoints_1, *ins[1], keypoints_2, matches, outs);
    drawMatches(*ins[0], keypoints_1, *ins[1], keypoints_2, good_matches, *outs);

    //imshow("After All Match", img_match);
    //imshow("Good Match", img_goodmatch);

    waitKey(0);
}

