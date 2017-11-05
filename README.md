This repo implement the algorithms used in image process.

Author don't guarantee the codes are all right. Just be examples.

1. ./nmf_matlab : the nmf factorization implemented based on matlab.
		
	Including files : *.m

2. ./rlnsw_matlab : the rlnsw implemented based on matlab
	
	rlnsw: redundant lifting non-seperable wavelet.

3. ./shearlet_matlab : the shearlet implemented based on matlab

	The shearlet tool can recur the coefficient matrix obtained by shearlet transform.

4. ./Insst_matlab : the improved non-subsampled shearlet transform
	
	Multiresulotion : RLNSW

	Directional     : Shearlet filters.

5. ./gauss_filter_cpp  : the 5*5 gauss filte based on c plus plus

	The first cpp file to do image process using opencv.

6. ./svd_opencv : calculate the svd of inputted image
	
	use '''convertTo ''' to convert the char data to float data.

	use '''svd::computer''' to calculate the SVD decompose.

7/ ./rlnsw_opencv_cpp : implementation of rlnsw in c plus plus

	rlnsw : redundant lifting non-seperable wavelet.

	usage :

		a. go to the build.

		b. cmake ..

		c. make 

		d. rlnsw filename.imagetype

		e. output the restored image obtained by rlnsw.

8. ./wavelet_opencv_cpp : implementation of two wavelets in c plus plus
	
	included : haar, sym2

	Usage :
		
		see the rlnsw's part.

9. ./improved rlnsw based on matlab

	add the two level rlnsw to be one level:
		horizontal&vertical predict + update
		+
		diagonal direction predict + update

	so, here the real level is 2^(input level)

	eg : [T, M, B] = rlnsw(V, 2);

	here the real level is equal to original version's 2^2 = 4 levels

10. ./nmf_opencv

	normal nmf algorithm based on opencv

		Usage:

			nmf  picture.name  rank  maxiter

	svd initialization  plus  multiplicative process

		failed, result cannot restore to the origin image, don't know reason ... 

		need more work ... 

	Latest news:

		I just FIXed the bugs leading to wrong restorage, and I removed the 'W / W_sum' statement 

11. ./svd_cuda_cpp

	image svd process based on NVIDIA CUDA

	Usage :

		svd image.name

	Description :
		
		Read image data with OpenCV and calculate SVD based on cusolver and cublas libraries .

		Time comsuption comparison between CUDA and OpenCV  

12 ./inmf_cuda_opencv_cpp

	image factorize based on improved nmf

	Usage : 

		svd image.name

	Description :

		read image data and then svd calculation

		after svd, use the results to initialize nmf and multiply update ...


13 ./HTO_GF_matlab

	hot-target-oriented image fusion based on guided filter

	usage :
		
		Res = fusion(A, B);

		A is visible image

		B is infrared image

	Data Type :

		double

	Description :

		Include Histogram calculation and image fusin based on histogram information 

		Then use guided filter to enhance the texture of fused image

14 ./histogram_cuda_cpp

	calculation of histogram of input image

	Usage :
		
		calHist(image.data, image size,distribution result);


	Data Type : 

		image data : unsigned char pointer to image data

		size 	   : const int 

		distribution result : unsigned int pointer, the distribution of result pointer : Range 0 ~ 256

	Description :

		with the help of addAtom, this function can calculate the image gray distribution in parallel way
		
		but, when the image is small, the parallel version is not faster than serial implementation by C++ array !

		if you will use this function many time, you can set this functioin to "inline"


15. ./sobel_opencv_cpp
	
	edge detector by sobel filter

16. ./histogram_fpga_hls

	This dir include needed fpga source files to implement the histogram calculation and contranst enhancement based on Xilinx HLS tools.

	The corresponding tutorial video can be found from https://www.youtube.com/watch?v=Po3KHF0SlIc&t=578s.

17. ./sobel_cuda_cpp

	This dir include the implementaton of sobel edge detection algorithm based on cuda. 

	//this is the first version, and the optimized version using texture memory is still under debugging...

	update : the second version using texture memory is uploaded. Usage: delete the origin sobel.cu and rename the sobel_optimized.cu to sobel.cu

18. ./strech_cuda_cpp

	this dir include a function to enhance the contrast of input image.
	
	very simple algorithm, so no more introduction.

19. ./NNDSVD_NMF_matlab

	This dir includes the implementation of NNDSVD-based initialization of NMF. 

	The Update rule is MM.

	Implement based on Matlab.

20. ./gff_matlab

	This dir includes the implementation of 'image fusion with guided filter' based on matlab. 

	More details about this fusion frame, please see the PDF paper.  Thanks to the paper authors.

	If you find errors, please let me know...

	Oops, the passwd for the zip file is my Github Name: TriLoo

	FreeBSD Copyright.

21. ./SRD_matlab

	This dir includes matlab files which realizes "Global Contrast based Salient Region Detection" HC algorithm, but only fit to gray scale input image. 
	
22. ./gff_OpenCV

	This dir include the implementation of 'image fusion with guided filter' based on OpenCV.

	More details can be found in corresponding paper.

	Caution : GPL Copyright. !!!

23. ./guidedFilter
	
	This dir implement the implementation of 'guided image filter' by He Kaiming etc.

	Pseudocode can be found in corresponding paper.

	It include two version: based on mean filter and box filter. Results show that the boxfilter can realize faster speed.

	Caution : GPL Copyright.

24. ./fastGuidedFilter

	This dir implement the implementation of 'fast guided image filter' by He Kaiming etc.

	Psedocode can be found in corresponding paper.

	Implement based on boxfilter and subsample.m and usample.m .

	The implementation of subsample.m and usample.m is referring to www.cnblogs.com/Allen-rg/p/5522412.html

	Caution : GPL Copyright.

25. ./boxfilter_cuda

	This dir include the files implementing the box filter based on cuda.

	Box filter is exactly the mean filter.

	Include : based on linear memory and array memory two versions.

26. ./ObjectTracking_opencv

	This dir include the practice based on Learning OpenCV Website

	So it is just calling some APIs and not involve any bottom implementation and algorithm structure
	as well.

27. ./NonPhotoRendering

	This dir include the practice based on Learning OpenCV Website

	So it is just calling some APIs and not involve any bottom implementation and algorithm struct 
	as well.

28. ./WeightMap_cuda_cpp

	This dir include the function of Weight Saliency Map generation based on " Image fusion 
	based on Guided Filter"

	The code is not tested for now ! ! !

29. ./BoxFilterFinal_CUDA

	This dir include the implementation of boxfilter based on four approaches:

		A. boxiflter based on separable row & col accumulate, the fastest version
		B. boxfilter based on shared memory                   the second fastest version
		C. boxiflter based on global memory                   the second slowest version
		D. boxfilter based on texture memory                  the slowest version

30. ./GuidedFilter_CUDA

	This dir include the implementatioin of guided filter based on CUDA

31. ./PCA_Python

	This dir include the implementation of PCA based on Python

32. ./ImageMosaicing_opencv

	This dir include the implementation of 'Image Mosaicing' based on ORB features.

	More details can see : http://www.cnblogs.com/skyfsm/p/7411961.html.
