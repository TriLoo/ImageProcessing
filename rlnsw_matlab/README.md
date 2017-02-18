This dir includes files to implement rlnsw based on matlab.

rlnsw : redundant lifting non-separable wavelet.

1. my_rlnsw.m : 

			Input  : image matrix

			Output : the low frequency matrix & high frequeny matrix 

			Description :
				copy - predict - update
				P = [1, 3/4; 2/3, 9/16];
				U = P/2;

2. inv_my_rlnsw.m
			Input  : highpass & lowpass matrix H, L
			Output : the matrix before RLNSW transform

			Description :
				get the image matrix.


3. Problem :
	After the RLNSW transform, the lowpass & highpass coefficient matrixs include negative numbers. 
