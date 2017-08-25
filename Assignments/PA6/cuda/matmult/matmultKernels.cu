///
/// matmultKernel00.cu
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernels.h"
#include <stdio.h>
#ifndef FOOTPRINT_SIZE
    #define FOOTPRINT_SIZE BLOCK_SIZE
#endif

__global__ void MatMul(double*__restrict__  A, double*__restrict__  B, double*__restrict__  C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
    double CValue = 0.0;
    
    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
        
        if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;
        
        if (k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        
        __syncthreads();
        
        for (int n = 0; n < BLOCK_SIZE; ++n)
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        
        __syncthreads();
    }
    
    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
        (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

__global__ void TMatMul(double*__restrict__  A, double*__restrict__  B, double*__restrict__  C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {
    double CValue = 0.0;
    
    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
        
        if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;
        
        if (k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        
        __syncthreads();
        
        for (int n = 0; n < BLOCK_SIZE; ++n)
            CValue += As[n][threadIdx.y] * Bs[n][threadIdx.x];
        
        __syncthreads();
    }
    
    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
        (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

__global__ void MatMulT(double*__restrict__  A, double*__restrict__  B, double*__restrict__  C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {
    double CValue = 0.0;
    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    int max = ((BLOCK_SIZE + ACols - 1)/BLOCK_SIZE);
    if(max < ((BLOCK_SIZE + BCols - 1)/BLOCK_SIZE))
        max = ((BLOCK_SIZE + BCols - 1)/BLOCK_SIZE);
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    for (int k = 0; k <max; k++) {
        
        if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;
        
        if (k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
        //if (k*BLOCK_SIZE + threadIdx.x < BCols && Row < BRows)    
            //Bs[threadIdx.y][threadIdx.x] = B[Row*BCols + k*BLOCK_SIZE + threadIdx.x];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        
        __syncthreads();
        
        for (int n = 0; n < BLOCK_SIZE; ++n)
            CValue += As[threadIdx.y][n] * Bs[threadIdx.x][n];
        
        __syncthreads();
    }
    
    if (Row < CRows && CValue != 0.0) {
        C[((blockIdx.y * blockDim.y + threadIdx.x)*CCols) +
        (blockIdx.x * blockDim.x)+ threadIdx.y] = CValue;
    }
}