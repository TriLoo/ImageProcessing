# 实验过程中的数据及其对比

smh

2018.12.08

*夫唯不争，故天下莫能与之争.*

## Kaptein 1654 IR

* 尺寸

  450 * 620
  
* 数据传输时间

  从Device到Host耗时约: 0.396ms

* 运行时间对比

  |   模块                |  CPU耗时(ms)  |  GPU耗时(ms)  |
  | :-----------------:  | :---------:  | :-------:   |
  |  Horizontal Predict  |  9.73305   |  1.20883  | 
  |  Horizontal Sinc     |  6.21211   |  0.37131  |
  |  Horizontal Update   |  8.54615   |  1.13098  |
  |  Horizontal P + U    |     -      |  2.33981  |
  |  Vertical P + U      |     -      |  ~1.4ms   |
  
  
* 总的耗时

  总的耗时包括了：数据传输(HtoD, DtoH)、分解算法(doRDLWavelet)、重构算法(doInverseRDLWavelet)等三个部分。
  
  经过CUDA Event测量，对于Kaptein 1654这幅图，CUDA实现的版本的耗时为：
  
  **7.5ms ~ 10.5ms**, 平均值为9ms(其实大部分比9ms更快). 截图如下：
  
  ![运行结果截图](RDLWavelet_CUDA0.png)


## Ball



## Source



## Lake
