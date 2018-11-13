
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include "pfilter-matrix.h"
#include "utils.h"
#include <math.h>
#include <omp.h>
#include "mex.h"

using namespace std;

/**
 * Main function of the propagation filter
 * @param OutPtr pointer to an array where result will be stored
 * @param RPtr pointer to an array containing reference matrix 
 * @param APtr pointer to an array containing input matrix 
 * @param w window radius
 * @param sigma_d
 * @param sigma_r
 * @param nRows the number of rows of R and A
 * @param nCols the number of columns of R and A
 * @param zR the number of channels of R
 * @param zA the number of channels of A
 */
void pfilter_matrix(double* OutPtr, const double* RPtr, const double* APtr, int w, double sigma_d, double sigma_r, int nRows, int nCols, int zR, int zA) {

    # ifdef OPENMP
    // set number of threads used by openmp
    // omp_set_num_threads(floor(omp_get_num_procs()/2));
    omp_set_num_threads(floor(omp_get_num_procs()-2));
    # endif

    #pragma omp parallel default(shared)
    {      
        double* resultPixelPtr = new double[zA];

        #pragma omp for schedule(runtime)
        for (int i = 0; i < nRows; ++i)
        {
            for (int j = 0; j < nCols; ++j)
            {
                int point[2] = {i, j};                
                calculateWeightedAverage(resultPixelPtr, RPtr, APtr, w, sigma_d, sigma_r, point, nRows, nCols, zR, zA);
                for (int z = 0; z < zA; ++z)
                {
                    getRef3D(OutPtr, i, j, z, nRows, nCols, zA) = resultPixelPtr[z];         
                }  
            }
        }        
        delete [] resultPixelPtr;
    }

    return;

}

/**
 * Calculate weighted average of pixels surrounding the point
 * @param resultPixelPtr the array (with the size = zA) contains the result pixel values 
 * @param RPtr
 * @param APtr
 * @param w
 * @param sigma_d
 * @param sigma_r
 * @param point
 * @param m the number of rows of R and A
 * @param n the number of columns of R and A
 * @return void
 */
void calculateWeightedAverage(double* resultPixelPtr, const double* RPtr, const double* APtr, int w, double sigma_d, double sigma_r, const int (&point)[2], int m, int n, int zR, int zA) {

    int wSize = 2 * w + 1;
    double* logWeightWindowPtr = new double[wSize * wSize];
    double totalWeight = 0;
    double* totalWeightedSum = new double[zA];
    for (int z = 0; z < zA; ++z)
    {
        totalWeightedSum[z] = 0;
    }
    
    // Calculate the weight of the Center Point
    getRef2D(logWeightWindowPtr, w, w, wSize, wSize) = 0;
    totalWeight += 1.0;
    for (int z = 0; z < zA; ++z)
    {
        totalWeightedSum[z] += getRef3D(APtr, point[0], point[1], z, m, n, zA);
    }

    // Calculate from distance 1 to w
    for (int r = 1; r < w + 1; ++r)
    {
        for (int dp = 0; dp < r + 1; ++dp)
        {
            for (int pSign = -1; pSign < 2; pSign += 2) // sign = -1, 1
            {
                int p = pSign*dp;


                for (int qSign = -1; qSign < 2; qSign += 2) // sign = -1, 1
                {
                    int q = qSign * (r - dp);

                    // check boundary
                    if (point[0] + p < 0 || point[0] + p > m - 1 || point[1] + q < 0 || point[1] + q > n - 1)
                    {
                        continue;
                    }

                    // decide fromLocal (the parent pixel t-1)
                    int fromLocal[2];
                    if (p * q == 0) // on the x or y axis
                    {
                        if (p == 0)
                        {
                            fromLocal[0] = p;
                            fromLocal[1] = q - qSign;
                        }
                        else // q == 0
                        {
                            fromLocal[0] = p - pSign;
                            fromLocal[1] = q;
                        }
                    }
                    else // p*q != 0 (other pixels)
                    {
                        // if r is odd -> p , else -> q
                        if (r % 2 != 0)
                        {
                            fromLocal[0] = p;
                            fromLocal[1] = q - qSign;
                        }
                        else
                        {
                            fromLocal[0] = p - pSign;
                            fromLocal[1] = q;
                        }
                    }

                    // calculate log weight
                    int toLocal[2] = {p, q};
                    double logWeight = calculateLogWeight(RPtr, sigma_d, sigma_r, logWeightWindowPtr, w, point, fromLocal, toLocal, m, n, zR);
                    getRef2D(logWeightWindowPtr, w + p, w + q, wSize, wSize) = logWeight;
                    double weight = exp(logWeight);               
                    
                    totalWeight += weight;
                    for (int z = 0; z < zA; ++z)
                    {
                        totalWeightedSum[z] += weight * getRef3D(APtr, point[0]+p, point[1]+q, z, m, n, zA);
                    }                    
                    
                    // ensure pixels on the axis is calculated only one time 
                    if (q == 0)
                    {
                        break;
                    }

                }

                // ensure pixels on the axis is calculated only one time 
                if (p == 0)
                {
                    break;
                }
            }
        }
    } 
    
    
    // Calculate result pixel value
    for (int z = 0; z < zA; ++z)
    {
        resultPixelPtr[z] = totalWeightedSum[z] / totalWeight;
    }    

    delete [] logWeightWindowPtr;
    delete [] totalWeightedSum;

}


/**
 * Calculate the log relationship of two points in R
 * @param RPtr
 * @param sigma
 * @param fromPoint
 * @param toPoint
 * @param nRows number of rows in R
 * @param nCols number of cols in R
 * @return 
 */
double calculateLogRelationship(const double* RPtr, double sigma, const int (&fromPoint)[2], const int (&toPoint)[2], int nRows, int nCols, int zR) {
    double distanceSquare = 0;
    for (int z = 0; z < zR; ++z)
    {
        double diff = getRef3D(RPtr, fromPoint[0], fromPoint[1], z, nRows, nCols, zR) - getRef3D(RPtr, toPoint[0], toPoint[1], z, nRows, nCols, zR);
        distanceSquare += pow(diff,2);
    }
    
    return -1 * distanceSquare / (2 * pow(sigma, 2));
}


/**
 * Calculate the log-weight between centerPoint and the toPoint through fromPoint
 * @param RPtr
 * @param sigma_d
 * @param sigma_r
 * @param fromPoint
 * @param toPoint
 * @param nRows number of rows in R
 * @param nCols number of cols in R
 * @return 
 */
double calculateLogWeight(const double* RPtr, double sigma_d, double sigma_r, const double* logWeightWindowPtr, int w, const int (&centerPoint)[2], const int (&fromLocal)[2], const int (&toLocal)[2], int nRows, int nCols, int zR) {

    int fromPoint[2] = {centerPoint[0] + fromLocal[0], centerPoint[1] + fromLocal[1]};
    int toPoint[2] = {centerPoint[0] + toLocal[0], centerPoint[1] + toLocal[1]};

    double pathLogProb = calculateLogRelationship(RPtr, sigma_d, fromPoint, toPoint, nRows, nCols, zR);
    double rangeLogProb = calculateLogRelationship(RPtr, sigma_r, centerPoint, toPoint, nRows, nCols, zR);
    double logWeight = getRef2D(logWeightWindowPtr, fromLocal[0] + w, fromLocal[1] + w, 2 * w + 1, 2 * w + 1) + pathLogProb + rangeLogProb;

    return logWeight;
}


