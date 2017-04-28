/*
 * calHist.h
 *
 *  Created on: Apr 28, 2017
 *      Author: smher
 */

#ifndef CALHIST_H_
#define CALHIST_H_

#include <iostream>

using namespace std;

#define GRID 256
#define BLOCK 256

/* informatioin about this function
 * Function : calculate the histogram distribution of image
 * Input :
 * 		data : the data pointer, locate on Host
 * 		size : the size of input data
 * 		hist : the histogram distribution pointer, locate on Host
 */
void calHist(unsigned char *data, const int size, unsigned int *hist);



#endif /* CALHIST_H_ */
