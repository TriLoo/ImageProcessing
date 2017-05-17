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

