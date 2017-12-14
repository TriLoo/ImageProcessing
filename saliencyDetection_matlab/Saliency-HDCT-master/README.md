# Saliency-HDCT
Matlab Implementation of the paper "Salient Region Detection via High-Dimensional Color Transform and Local Spatial Support" 

https://sites.google.com/site/kjw02040/hdct

## Abstract
In this paper, we introduce a novel technique to automatically detect salient regions of an image via high-dimensional color transform. Our main idea is to represent a saliency map of an image as a linear combination of high-dimensional color space where salient regions and backgrounds can be distinctively separated. This is based on an observation that salient regions often have distinctive colors compared to the background in human perception, but human perception is often complicated and highly nonlinear. By mapping a low dimensional RGB color to a feature vector in a high-dimensional color space, we show that we can linearly separate the salient regions from the background by finding an optimal linear combination of color coefficients in the high-dimensional color space. Our high dimensional color space incorporates multiple color representations including RGB, CIELab, HSV and with gamma corrections to enrich its representative power. Our experimental results on three benchmark datasets show that our technique is effective, and it is computationally efficient in comparison to previous state-of-the-art techniques.

Note : Our code requires the VLfeat library, which can be downloaded at : http://www.vlfeat.org/

## Usage

0. Install required libraries and compile :
 1. VLfeat (http://www.vlfeat.org/)
 2. Histogram distance toolbox (http://www.mathworks.com/matlabcentral/fileexchange/39275-histogram-distances)
 3. SQBlib (http://sites.google.com/site/carlosbecker/)
1. Save the test images at 'images' folder.
2. Run main.m

We tested our code in Matlab 2015a, Windows 7 environment.  Lower version of Matlab may cause several errors.

## Results

<img src="https://cloud.githubusercontent.com/assets/22743125/19303504/df0bc786-90a3-11e6-9058-05dfe89c765e.png" width="960">
Figure 1. Visual comparisons of our results and results from previous methods.  Each image denotes (a) test image, (b) ground truth, (c) our approach.  (d) DRFI [1], (e) GMR [2], (f) HS[3], (g) SF[4], (h) LR[5], (i) RC[6], (j) HC[6], (k) LC[7].


## Paper

Please cite one of these papers if you use this code:

0. **Jiwhan Kim**, Dongyoon Han, Yu-Wing Tai, and Junmo Kim, "Salient Region Detection via High-Dimensional Color Transform and Local Spatial Support", IEEE Transactions on Image Processing, Vol. 25, No. 1, pp. 9-23, Jan. 2016.
1. **Jiwhan Kim**, Dongyoon Han, Yu-Wing Tai, and Junmo Kim, "Salient Region Detection via High-Dimensional Color Transform", CVPR 2014.

## References
0. H. Jiang, J. Wang, Z. Yuan, Y. Wu, N. Zheng, and S. Li, " Salient object detection: A discriminative regional feature integration approach", CVPR, 2013.
1. C. Yang, L. Zhang, H. Lu, X. Ruan, and M.-H. Yang, "Saliency detection via graph-based manifold ranking", CVPR, 2013.
2. Q. Yan, L. Xu, J. Shi, and J. Jia, "Hierarchical saliency detection", CVPR, 2013.
3. F. Perazzi, P. Krahenbuhl, Y. Pritch, and A. Hornung, "Saliency filters: Contrast based filtering for salient object detection", CVPR, 2012.
4. X. Shen, and Y. Wu, "A unified approach to salient object detection via low rank matrix recovery", CVPR, 2012.
5. M. Cheng, G. Zhang, M. Mitra, X. Huang, and S. Hu, "Global contrast based salient region detection", CVPR, 2011.
6. Y. Zhai, and M. Shah, "Visual attention detection in video sequences using spatiotemporal cues", ACM Multimedia, 2006.
