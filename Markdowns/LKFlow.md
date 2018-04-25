# LK Flow Notes

## 公式

$$ E_x u + E_y v + E_t = 0 $$

其中，$E_x=∂E/∂x$，其它的同理。

## 改进

引入金字塔分层，这样可以保证图像中物体的运动速度较小。

### 步骤

* 建立金字塔

    下采样过程：用一个Gaussian低通滤波器对$L^{L-1}$进行卷积, 步长为1。

* 金字塔跟踪

* 迭代

## 代码阅读

[CSND LK](https://blog.csdn.net/u014568921/article/details//46638557)

* void lowpass_filter(BYTE *&src, const int H, const int W, BYTE *&smoothed)

    也就是利用Gaussian Filter对src图像进行低通滤波，结果保存在smoothed里面。

* get_info(const int nh, const int nw)

    产生保存每一层图像的尺寸的数组

* get_target()

    设置某一点的x、y分量。

* BYTE* get_pyramid(int nh)

    返回第nh个Pyramid分层图像

* Point get_result()

    返回target[0]的x、y两个分量

* void get_pre_frame(BYTE* &gray)

    build_pyramid(pre_pyr)

* void discard_pre_frame()

    delete [] pre_pyr[i]

* void get_pre_frame()
