#ifndef __FUNCKERNELS__
#define __FUNCKERNELS__
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

__global__ void function(double *__restrict__ A, double * __restrict__ C, int ARows, int ACols, int CRows, int CCols, long val);
__global__ void gradient_function(double * __restrict__ A, double * __restrict__ C, int ARows, int ACols, int CRows, int CCols);
__global__ void error_function(double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols);
__global__ void reduction_function(double * __restrict__ A, double * __restrict__ C, int ARows, int ACols, int Clength);
__global__ void normalize(double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int ARows, int ACols, int Blength, int CRows, int CCols);
__global__ void delta_function(double* __restrict__ A, double* __restrict__ C, int ARows, int ACols, int CRows, int CCols, double val);
#endif // __FUNCKERNELS__