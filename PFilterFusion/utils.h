/* 
 * File:   utils.h
 * Author: rick
 *
 * Created on December 5, 2013, 1:10 PM
 */

#ifndef UTILS_H
#define	UTILS_H

// access matlab 2D matrix
#define getRef2D(dataPtr,i,j,m,n) dataPtr[(i)+(j)*(m)]

// access matlab 3D matrix
#define getRef3D(dataPtr,i,j,k,m,n,z) dataPtr[(i)+((j)+(k)*(n))*(m)]


#endif	/* UTILS_H */

