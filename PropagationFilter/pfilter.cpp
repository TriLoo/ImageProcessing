
#include "mex.h"
#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include "pfilter-matrix.h"
#include "utils.h"

using namespace std;

/**
 * The gateway between this program and MATLAB
 * Usage: result = pfilter(reference, input, w, sigma);
 * 
 * Input arguments
 * reference:   Guidance image from which the weights are calculated.  This can be the input image itself or other images.  
 *              It is a m*n*c matrix, where c can be 1 (gray-scale image) or 3 (colorful image).
 *              The reference image can use any color space.  Distance between two pixel values are calculated by Euclidean distance.
 * input:       The input image, from which its pixel values are averaged.  
 *              It is a m*n*c matrix, where c can be 1 (gray-scale image) or 3 (colorful image).
 * w:           Window radius
 * sigma:       1*2 matrix [sigma_d sigma_r]
 * sigma_d:     Sigma for the Gaussian function determining the relationship between two adjacent pixels t-1 and t
 * sigma_r:     Sigma for the Gaussian function determining the relationship between the centered pixel s and the pixel t 
 *     
 * Output arguments
 * result:      Filtered image (m*n*c matrix)       
 * 
 * Note
 * The reference image and the input image can in different color space.  
 * For example, the reference image can be a m*n*3 color image, while the input image can be a m*n gray-scale image.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // check for the proper number of arguments
    if (nrhs != 4)
        mexErrMsgIdAndTxt("MATLAB:pfilter:invalidNumInputs",
            "Input: reference(m*n*c matrix) input(m*n*c matrix) w [sigma_d sigma_r]");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("MATLAB:pfilter:maxlhs",
            "Too many output arguments.");

    // Check that input and reference are m*n*c matrix
    if (mxGetNumberOfDimensions(prhs[0]) > 3 || mxGetNumberOfDimensions(prhs[1]) > 3)
        mexErrMsgIdAndTxt("MATLAB:pfilter:wrongInputFormat",
            "Input and Reference should be m*n or m*n*3 matrix.");
    if (mxGetNumberOfDimensions(prhs[0]) == 3 && ((mxGetDimensions(prhs[0]))[2] != 1 && (mxGetDimensions(prhs[0]))[2] != 3))
        mexErrMsgIdAndTxt("MATLAB:pfilter:wrongInputFormat",
            "Reference should be m*n or m*n*3 matrix.");
    if (mxGetNumberOfDimensions(prhs[1]) == 3 && ((mxGetDimensions(prhs[1]))[2] != 1 && (mxGetDimensions(prhs[1]))[2] != 3))
        mexErrMsgIdAndTxt("MATLAB:pfilter:wrongInputFormat",
            "Input should be m*n or m*n*3 matrix.");

    // Check that input and reference have the same size (Row, Col) 
    if ((mxGetDimensions(prhs[0]))[0] != (mxGetDimensions(prhs[1]))[0] || (mxGetDimensions(prhs[0]))[1] != (mxGetDimensions(prhs[1]))[1])
        mexErrMsgIdAndTxt("MATLAB:pfilter:wrongInputFormat",
            "Input and Reference should have the same m and n.");
    
    // Check that sigma = [sigma_d sigma_r]
    if (mxGetNumberOfElements(prhs[3]) != 2)
        mexErrMsgIdAndTxt("MATLAB:pfilter:wrongInputFormat",
            "Sigma = [Sigma_d Sigma_r]");

    
    
    mwSize m = (mxGetDimensions(prhs[0]))[0];
    mwSize n = (mxGetDimensions(prhs[0]))[1];
    mwSize zRef, zInput;
    if (mxGetNumberOfDimensions(prhs[0]) == 2)
        zRef = 1;
    else
        zRef = (mxGetDimensions(prhs[0]))[2];

    if (mxGetNumberOfDimensions(prhs[1]) == 2)
        zInput = 1;
    else
        zInput = (mxGetDimensions(prhs[1]))[2];
    
    
    mwSize w = (mwSize) mxGetScalar(prhs[2]);
    double sigma_d = mxGetPr(prhs[3])[0];
    double sigma_r = mxGetPr(prhs[3])[1];


    // Allocate output    
    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]), mxGetDimensions(prhs[1]), mxDOUBLE_CLASS, mxREAL);
    double* resultPtr = mxGetPr(plhs[0]);
    
    // Get reference matrix and input matrix data pointers
    const double* referencePtr = mxGetPr(prhs[0]);
    const double* inputPtr = mxGetPr(prhs[1]);
    
//    clock_t start = clock();
    pfilter_matrix(resultPtr, referencePtr, inputPtr, (int)w, sigma_d, sigma_r, (int)m, (int)n, (int)zRef, (int)zInput);
//    clock_t end = clock();
//    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//    printf("Use %f seconds on %d * %d matrix.\n", cpu_time_used, m, n);

    return;
}





