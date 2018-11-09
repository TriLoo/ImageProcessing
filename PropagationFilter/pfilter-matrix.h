/* 
 * File:   sfilter-matrix.h
 * Author: rick
 *
 * Created on December 5, 2013, 1:09 PM
 */

#ifndef SFILTER_MATRIX_H
#define	SFILTER_MATRIX_H

void pfilter_matrix(double* OutPtr, const double* RPtr, const double* APtr, int w, double sigma_d, double sigma_r, int nRows, int nCols, int zR, int zA);
void calculateWeightedAverage(double* resultPixelPtr, const double* RPtr, const double* APtr, int w, double sigma_d, double sigma_r, const int (&point)[2], int m, int n, int zR, int zA);
double calculateLogRelationship(const double* RPtr, double sigma, const int (&fromPoint)[2], const int (&toPoint)[2], int nRows, int nCols, int zR);
double calculateLogWeight(const double* RPtr, double sigma_d, double sigma_r, const double* logWeightWindowPtr, int w, const int (&centerPoint)[2], const int (&fromLocal)[2], const int (&toLocal)[2], int nRows, int nCols, int zR);
#endif	/* SFILTER_MATRIX_H */

