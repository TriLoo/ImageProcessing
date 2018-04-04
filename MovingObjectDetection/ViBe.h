//
// Created by smher on 18-4-4.
//

#ifndef MOVINGOBJECTDETECTS_VIBE_H
#define MOVINGOBJECTDETECTS_VIBE_H

#include "headers.h"

class  ViBe
{
public:
    ViBe(int n = 20, int r = 20, int t = 2, int s = 16, int i = 0);
    //ViBe(int n, int r, int t, int s, int i);

    void initViBe(int wid, int hei);
    void detectionBG(cv::Mat& imgIn);
    void initialFrame(cv::Mat& imgIn);
    cv::Mat getBGimg()
    {
        return imgBG_;
    }

private:
    std::vector<cv::Mat> bgModels_;
    std::vector<int> neiPos_;   // a 1 * 9 vector, storing the position of neighboring 3 * 3 elements
    cv::Mat imgBG_;
    cv::Mat fgTimes_;

    int num_samp_;   // 样本点值N
    int R_, T_;      // R: 距离阈值，T: 判断前景点的阈值
    int samp_rate_;  // 采样概率

    int id_;         // 存储当前帧的数量
};

#endif //MOVINGOBJECTDETECTS_VIBE_H
