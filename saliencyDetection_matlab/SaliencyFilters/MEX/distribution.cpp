#define POSITION_DIM 3
#define VALUE_DIM	 4

#include "mex.h"
#include "filter.h"
#include <math.h>
#include <vector>
using std::vector;

void mexFunction ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	double *centers, *D;	//center positions of each superpixel and the result 
	double *w;				//Gaussian weight of position
	int length_centers ;
	double temp;
	double ix, iy;			//the x y of i
	double jx, jy;			//the x y of j


	//set the pointer of input & output

	centers = mxGetPr(prhs[0]);
	length_centers = mxGetM(prhs[0]);
	w = new double[ length_centers ] ;

	double *center_color ;
	double color , position,  ci, cj, mu;
	center_color = mxGetPr(prhs[1]);
	const double sigma = mxGetScalar(prhs[2]); 

	plhs[0]=mxCreateDoubleMatrix(length_centers,1,mxREAL);
	D = mxGetPr(plhs[0]) ;

	// Construct the 2-dimensional position vectors and
	// four-dimensional value vectors
	vector<float> positions(length_centers*POSITION_DIM);
	vector<float> values(length_centers*4);
	for ( int idx = 0 ; idx < length_centers ; idx ++ ) {
		positions[idx*POSITION_DIM+0] = *(center_color + idx + 0*length_centers)/sigma;
		positions[idx*POSITION_DIM+1] = *(center_color + idx + 1*length_centers)/sigma;
		positions[idx*POSITION_DIM+2] = *(center_color + idx + 2*length_centers)/sigma;
		values[idx*VALUE_DIM+0] = *(centers + idx );
		values[idx*VALUE_DIM+1] = *(centers + idx + length_centers );
		values[idx*VALUE_DIM+2] = pow(*(centers + idx ),2) + pow(*(centers + idx + length_centers ),2);
		values[idx*VALUE_DIM+3] = 1.0f;
	}

	// Perform the Gauss transform. For the 2-dimensional case the
	// Permutohedral Lattice is appropriate.
	Filter distribution( &positions[0], length_centers, POSITION_DIM);
	distribution.filter( &values[0], &values[0], VALUE_DIM ) ;

	// Divide through by the homogeneous coordinate and store the
	// result back to the image
	for (int i = 0; i < length_centers; i++) {
		*( D + i ) = 0 ;
		float w = values[i*VALUE_DIM+VALUE_DIM-1];
		for ( int j = 0 ; j < 2 ; j ++) {
			*( D + i ) -= pow(values[i*VALUE_DIM+j]/w, 2);
		}
		*( D + i ) += values[i*VALUE_DIM+2]/w  ;
	}
}