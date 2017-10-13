#include <iostream>
#include "vector"
#include "chrono"
#include "stdexcept"
#include "boost/thread.hpp"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;

// keypoints of input images
void ROBExtract(vector<vector<KeyPoint>> &keypoints, vector<Mat *> &descriptors,  vector<Mat *> &inImgs)
{
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    cout << "Image nums: " << inImgs.size() << endl;
    // Step 1 : 检测Oriented Fast角点位置
    //boost::thread orbThread(orb->detect, boost::ref(*inImgs[0]), boost::ref(keypoints[0]));
    //boost::thread orbThread(orb->detect, *inImgs[0], keypoints[0]);
    orb->detect(*inImgs[0], keypoints[0]);
    orb->detect(*inImgs[1], keypoints[1]);
    //orbThread.join();

    // Step 2 : 计算描述子
    orb->compute(*inImgs[0], keypoints[0], *descriptors[0]);
    orb->compute(*inImgs[1], keypoints[1], *descriptors[1]);

    Mat outimg1;
    drawKeypoints(*inImgs[0], keypoints[0], outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB特征点", outimg1);

    // Step 3 : 对两幅图像中的BRIEF描述子进行匹配
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(*descriptors[0], *descriptors[1], matches);

    // Step 4 : 匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    // 找出所有匹配点之间的最小距离和最大距离，
    for(int i = 0; i < (*descriptors[0]).rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }

    cout << "Max dist : " << max_dist << endl;
    cout << "Min dist : " << min_dist << endl;

    vector<DMatch> good_matches;
    for(int i = 0; i < (*descriptors[0]).rows; i++)
    {
        if(matches[i].distance < max(2 * min_dist, 30.0))
            good_matches.push_back(matches[i]);
    }

    cout << "Origin size : " << matches.size() << endl;
    cout << "Good size : " << good_matches.size() << endl;

    // Step 5 : 显示匹配结果
    Mat img_match, img_goodmatch;
    drawMatches(*inImgs[0], keypoints[0], *inImgs[1], keypoints[1], matches, img_match);
    drawMatches(*inImgs[0], keypoints[0], *inImgs[1], keypoints[1], matches, img_goodmatch);

    imshow("Origin Match", img_match);
    imshow("Good Match", img_goodmatch);

    waitKey(0);
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat imgA = imread("1.png", IMREAD_COLOR);
    Mat imgB = imread("2.png", IMREAD_COLOR);

    if(!imgA.data || !imgB.data)
    {
        cerr << "Read image data failed ..." << endl;
        return -1;
    }

    Mat descriptor_1, descriptor_2;
    vector<vector<KeyPoint>> keypoints(2);
    vector<Mat *> descriptors = {&descriptor_1, &descriptor_2};
    vector<Mat *> inImgs = {&imgA, &imgB};

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ROBExtract(keypoints, descriptors, inImgs);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time used to extract features : " << time_used.count() << " s" << endl;

    waitKey(0);

    return 0;
}
