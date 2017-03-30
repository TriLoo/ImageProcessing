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
