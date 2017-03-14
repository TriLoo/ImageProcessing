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

