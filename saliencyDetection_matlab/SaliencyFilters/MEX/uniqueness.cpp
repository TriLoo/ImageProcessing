#define POSITION_DIM 2
#define VALUE_DIM	 5


#include "mex.h"
#include "filter.h"
#include <math.h>
#include <vector>
using std::vector;

void mexFunction ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	double *centers, *U;	//center positions of each superpixel and the result 
	double *w;				//Gaussian weight of position
	int length_centers;

	//set the pointer of input & output

	centers = mxGetPr(prhs[0]);
	length_centers = mxGetM(prhs[0]);
	w = new double[length_centers] ;

	double *center_color ;
	double color_i, color_j , ci, cj, color;
	center_color = mxGetPr(prhs[1]);
	

	const double sigma = mxGetScalar(prhs[2]); 
	//double sigma = 25;

	plhs[0]=mxCreateDoubleMatrix(length_centers,1,mxREAL);
	U = mxGetPr(plhs[0]) ;

	// Construct the 2-dimensional position vectors and
	// four-dimensional value vectors
	vector<float> positions(length_centers*POSITION_DIM);
	vector<float> values(length_centers*VALUE_DIM);
	for ( int i = 0 ; i < length_centers ; i ++ ) {
		positions[i*POSITION_DIM+0] = *(centers + i )/sigma;
		positions[i*POSITION_DIM+1] = *(centers + i + length_centers )/sigma;
		values[i*VALUE_DIM+0] = *(center_color + i + 0*length_centers);
		values[i*VALUE_DIM+1] = *(center_color + i + 1*length_centers);
		values[i*VALUE_DIM+2] = *(center_color + i + 2*length_centers);
		values[i*VALUE_DIM+3] = pow(*(center_color + i + 0*length_centers),2)
							  + pow(*(center_color + i + 1*length_centers),2)
							  + pow(*(center_color + i + 2*length_centers),2);
		values[i*VALUE_DIM+4] = 1.0f;
	}

	// Perform the Gauss transform. For the 2-dimensional case the
	// Permutohedral Lattice is appropriate.
	Filter uniqueness( &positions[0], length_centers, POSITION_DIM);
	uniqueness.filter( &values[0], &values[0], VALUE_DIM ) ;

	// Divide through by the homogeneous coordinate and store the
	// result back to the image
	for (int i = 0; i < length_centers; i++) {
		float w = values[i*VALUE_DIM+VALUE_DIM-1];
		*(U + i ) = 0 ; 
		ci = 0 ;
		for ( int j = 0 ; j < 3 ; j ++) {
			color_i = *(center_color + i + length_centers*j)  ;
			ci += color_i * color_i ;
			*( U + i ) -= 2*values[i*VALUE_DIM+j]*color_i;
		}
		*( U + i ) += values[i*VALUE_DIM+3] + ci*w ;
	}
}
