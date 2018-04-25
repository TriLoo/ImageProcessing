# Redundant Directional Lifting-based Wavelet Optimization

## Steps

### Step 1

* Horizontal Sinc Iterpolation
* Horizontal Prediction
* Horizontal Sinc Iterpolation
* Horizontal Update
- - - - - -

### Step 2

* Transpose
* Horizontal Sinc Iterpolation
* Horizontal Prediction
* Horizontal Sinc Iterpolation
* Horizontal Update
- - - - - -

### Step 3

* Split

## 公式

* Sinc Interpolation

  $$ sum1 += inData(i, x) * Sinc(1, l + mySample) $$

  $$ sum2 += inData(i, x) * Sinc(2, l + mySample) $$

  $$ sum3 += inData(i, x) * Sinc(3, ll + mySample) $$

  **Out[1:4]**=[inData(i, j), $sum1$, $sum2$, $sum3$]

* Horizontal Prediction

  $$ tempSum += originT(i - 1, 4 * j - 3  + k + Dir) + originT(i+1, 4 * j -3  + Dir + k); $$

  $$originNew(i - Dir, j) = origin(i - Dir, j) - (tempSum /Divd)$$

* Horizontal Update

  $$ tempSum += originT(i - 1, 4 * j - 3  + k + Dir) + originT(i+1, 4 * j -3  + Dir + k); $$

  $$originNew(i - Dir, j) = origin(i - Dir, j) +  (tempSUm /Divd)$$

* Transpose

$Data(i, j) = Data(j , i)$

## Optimization

### Sinc Interpolation

插值的过程，本质上就是一个对输入图像进行三次的*一维卷积*操作，所以有以下初步实现方案：

* 输入图像为原始图像
* 输出图像为三个分离的图像，大小与输入图像同，这样可以方便索引的计算，从而方便存储与读取
* 下一步是优化一维卷积的操作，现在想到的就是：把三个一维卷积在一个Kernel函数里面实现，这样可以消除多余的不必要的对输入图像的访存，但还需要进一步优化实现

* 假设得到的矩阵分别记为：inA, inB, inC, inD，其中inA为原始输入图像，inB、inC、inD分别对应Sum1, Sum2, Sum3的数值

- - - - - -

### Horizontal Prediction

这一步需要计算18个数的和，分为上层9个、下层9个。上层的中，从输入图像中周围三个像素 + 另外三个矩阵的6个像素，分别在Z方向上是正上方一个像素 + 左上方一个像素。

初步想法就先这样吧