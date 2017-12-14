Name:
Saliency Filters

Reference:
Perazzi, Federico, et al. "Saliency filters: Contrast based filtering for salient region detection." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.

Implemented by 
Yifeng Zhang
Tianjin University, China

Note:
1. The result of super-pixel named 'sp' should be a double matrix
2. any super-pixel methods is OK, but the mask of superpixel MUST BE double type
3. If you have any question, please visit http://blog.csdn.net/xuanwu_yan/article/details/7734173 and reply, I'm pretty glad to answer it.
4. If you release any improvement of this, please keep my name on the thanks part :D

How to use:
1. mex -setup(if you didn't use mex before)
2. cd the \SaliencyFilters\MEX dictionary and run make_all.m
3. cd the \SaliencyFilters dictionary, run demo_saliency.m
4. if you have a trouble in mexGenerateSuperPixel.mexw64, please use other superpixel methods instead.

