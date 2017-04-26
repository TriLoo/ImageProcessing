/*
 * matrix.h
 *
 *  Created on: Apr 11, 2017
 *      Author: smher
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cublas.h>
#include <math.h>
#include <sys/time.h>


typedef struct{
    float* mat;  //pointer to host data
    float* mat_d; //pointer to device data
    int dim[2];   //dimensions: {rows,cols}
} matrix;

typedef enum{
    compute,
    cleanup
} action_t;

//creating, allocating, moving matrices
void read_matrix(matrix* A, char* file);
void write_matrix(matrix A, char* file);
void create_matrix(matrix* A, int rows, int cols, float value);
void create_matrix_on_device(matrix* A, int rows, int cols, float value);
void create_matrix_on_both(matrix* A, int rows, int cols, float value);
void copy_matrix_to_device(matrix* A);
void copy_matrix_on_device(matrix A, matrix B);
void copy_matrix_from_device(matrix* A);
void copy_to_padded(matrix A, matrix Apad);
void copy_matrix_to_device_padded(matrix A, matrix Apad);
void copy_from_padded(matrix A, matrix Apad);
void copy_matrix_from_device_padded(matrix A, matrix Apad);
void allocate_matrix_on_device(matrix* A);
void free_matrix_on_device(matrix* A);
void destroy_matrix(matrix* A);

//matrix analysis
void print_matrix(matrix A);
float matrix_difference_norm_d(action_t action,  matrix a, matrix c, int* params);
float matrix_div_d(action_t action, matrix a, matrix b, int* params);
float nan_check_d(action_t action, matrix a, int* params);
float zero_check_d(action_t action, matrix a, int* params);
float zero_check(matrix a);

//sgemms
void matrix_multiply_d( matrix a, matrix b, matrix c );
void matrix_multiply_AtB_d( matrix a, matrix b, matrix c );
void matrix_multiply_ABt_d( matrix a, matrix b, matrix c );

//element operations
void element_multiply_d( matrix a, matrix b, matrix c, int block_size);
void element_divide_d( matrix a, matrix b, matrix c, int block_size);
void matrix_eps_d( matrix a, int block_size);
void matrix_eps(matrix a);

//row/col-wise
void row_divide_d( matrix a, matrix b, matrix c);
void col_divide_d( matrix a, matrix b, matrix c);
void sum_cols_d(action_t action, matrix a, matrix c, int* params);
void sum_rows_d(action_t action, matrix a, matrix c, int* params);

// add four function :
// 		0.	choose the rank  #DONE#
// 		1. 	initialize matrix by a range of another matrix
// 		2. 	restore a matrix from vector
// 		3. 	calculate the absolute value of matrix   #DONE#
// then, transform the results to update functions


// choose the rank
int choose_rank(float *a, int m);   // m is the size of *a

// calcuate the abs of matrix
void mat_abs_d(matrix &a);

// restore a vector to a matrix
void mat_Restore(matrix &a, float *data, int m); // m is the size of data

// reshape a matrix to row * col
void mat_Reshape(matrix &a,float *data, int rowA, int colA, int colData);




#endif /* MATRIX_H_ */
