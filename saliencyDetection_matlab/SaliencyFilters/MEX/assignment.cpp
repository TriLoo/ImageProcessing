#define POSITION_DIM 5
#define VALUE_DIM	 4

#include "mex.h"
#include "filter.h"
#include <cmath>
#include <vector>
using std::vector;

enum INPUTARRAY
{
	UNIQUENESS, DISTRIBUTION, CENTRES, IMAGE, CENTRECOLOR, MASK, ALPHA, BETA, KEAY
};

double Sum_S( double *s, int size ) {
	double summ = 0;
	for ( int i = 0 ; i < size ; i ++) {
		summ += s[i];
	}
	return summ;
}

double max(double a,double b)
{
	return a>b?a:b;
}

double min(double a,double b){
	return a>b?b:a;
}

void mexFunction ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	double *U, *D, *centers, *img, *center_color, *masks;	
	double *S;				//final result and Gaussian weight

	int length_centers ;			//number of superpixel, as the length of uniqueness result and distribution result, and
							//also the number of centers
	double temp;
	int i;
	int ix, iy;			//the x y of i
	double jx, jy;			//the x y of j
	int mask_i;


	//set the pointer of input & output

	U = mxGetPr(prhs[UNIQUENESS]) ;			//uniqueness 

	D = mxGetPr(prhs[DISTRIBUTION]) ;			//distribution

	centers = mxGetPr(prhs[CENTRES]) ;
	length_centers = (int) mxGetM(prhs[CENTRES]) ;

	img =  mxGetPr(prhs[IMAGE]) ;
	const int *size_img = mxGetDimensions(prhs[IMAGE]) ;
	int x_img = (int) size_img[0] ;
	int y_img = (int) size_img[1] ;
	int z_img = (int) size_img[2] ;
	int image_size = x_img*y_img ;
	double color , ci, cj;
	int postition;
	//mexPrintf("the size is %d %d %d", x_img, y_img, z_img);

	center_color = mxGetPr(prhs[CENTRECOLOR]);

	masks = mxGetPr(prhs[MASK]);
	const int *size_mask = mxGetDimensions(prhs[MASK]) ;
	double alpha = *(mxGetPr(prhs[ALPHA]));				//position
	double beta = *(mxGetPr(prhs[BETA])) ;				//color
	double k = *(mxGetPr(prhs[KEAY])) ;

	//the output
	plhs[0]=mxCreateDoubleMatrix(x_img, y_img, mxREAL);
	S = mxGetPr(plhs[0]) ;
	double *temp_S = new double[length_centers] ;

	//because U and D are normalized in matlab, so we start from the next part.
	//in the part we calculate the temp_S first
	for ( i = 0 ; i < length_centers ; i ++) {
		temp_S[i] = U[i] * exp( -k * D[i]) ;
		/*mexPrintf("the temp_S is %f\n", temp_S[i]);*/
	}
	//mexPrintf("Hello! World!\n");
	// Construct the 2-dimensional position vectors and
	// four-dimensional value vectors
	vector<float> source_positions(image_size*POSITION_DIM);
	vector<float> value(image_size*VALUE_DIM);
	vector<float> targer_positions(image_size*POSITION_DIM);

	int seg_id ;
	for ( int idx = 0 ; idx < image_size ; idx ++ ) {
		seg_id = *(masks + idx ) ;
		source_positions[idx*POSITION_DIM+0] = idx / x_img * alpha ;
		source_positions[idx*POSITION_DIM+1] = idx % x_img * alpha ;
		source_positions[idx*POSITION_DIM+2] = *(center_color + seg_id + 0*length_centers)*beta;
		source_positions[idx*POSITION_DIM+3] = *(center_color + seg_id + 1*length_centers)*beta;
		source_positions[idx*POSITION_DIM+4] = *(center_color + seg_id + 2*length_centers)*beta;
		value[idx*VALUE_DIM+0] = temp_S[seg_id];
		value[idx*VALUE_DIM+1] = 1.0f;
	}

	for ( int idx = 0 ; idx < image_size ; idx ++ ) {
		targer_positions[idx*POSITION_DIM+0] = idx / x_img * alpha ;
		targer_positions[idx*POSITION_DIM+1] = idx % x_img * alpha ;
		targer_positions[idx*POSITION_DIM+2] = *(img + idx + 0*image_size)*beta;
		targer_positions[idx*POSITION_DIM+3] = *(img + idx + 1*image_size)*beta;
		targer_positions[idx*POSITION_DIM+4] = *(img + idx + 2*image_size)*beta;

	}
	// Perform the Gauss transform. For the 2-dimensional case the
	// Permutohedral Lattice is appropriate.
	Filter assignment( &source_positions[0], image_size, &targer_positions[0], image_size,  POSITION_DIM);
	assignment.filter( &value[0], &value[0], VALUE_DIM ) ;

	float w ;
	// Divide through by the homogeneous coordinate and store the
	// result back to the image
	for (int i = 0; i < image_size; i++) {
		w = value[ i * VALUE_DIM + VALUE_DIM -1];
		S[i] = value[ i * VALUE_DIM]/(w+1e-10)  ;
	}

	double max_S = 0, min_S = 1 ;
	for (int i = 0; i < image_size; i++) {
		max_S = max( S[i], max_S ) ;
		min_S = min( S[i], min_S ) ;
	}

	for (int i = 0; i < image_size; i++) {
		S[i] = (S[i] - min_S ) / ( max_S - min_S);
	}

	double m_sal = 0.1 * image_size;
	for( double sm = Sum_S( S, image_size ); sm < m_sal; sm = Sum_S( S, image_size ) ){
		for (int i =  0 ; i < image_size ; i++) {
			S[i] = min(S[i]*m_sal/sm, 1) ;
		}
	}
}

