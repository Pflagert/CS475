#include "functionKernels.h"
#define foo(a,b) b?tanh(a):exp(a)

__global__ void function(double * __restrict__ A, double * __restrict__ C, int ARows, int ACols, int CRows, int CCols, long val) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ARows || i >= CRows || j >= ACols || j >= CCols)
        return;
    int Cindex = (i * CCols) + (j + val);
    int Aindex = (i * ACols) + j;
    C[Cindex] = foo(A[Aindex], val);
}

__global__ void gradient_function(double * __restrict__ A, double * __restrict__ C, int ARows, int ACols, int CRows, int CCols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ARows || i >= CRows || j > ACols || j >= CCols)
        return;
    int Cindex = (i * CCols) + j;
    int Aindex = (i * ACols) + (j+1);
    C[Cindex] = A[Aindex] * (1 - pow( tanh(C[Cindex]), 2) );
}

__global__ void error_function(double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ARows || i >= BRows || i >= CRows 
        || j >= ACols || j >= BCols || j >= BCols) return;
    
    int Cindex = (i * CCols) + j;
    int Aindex = (i * ACols) + j;
    int Bindex = (i * BCols) + j;
    C[Cindex] = A[Aindex] - B[Bindex];
}

__global__ void reduction_function(double * __restrict__ A, double * __restrict__ C, int ARows, int ACols, int Clength) {
    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    As[threadIdx.y][threadIdx.x] = 0.0;
    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
        
        if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows) {
            As[threadIdx.y][threadIdx.x] += A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
            //printf("%d += A[%d]\n", threadIdx.x, Row*ACols + k*BLOCK_SIZE + threadIdx.x);
        }
        
        __syncthreads();
    }
    
    if (Row < Clength && threadIdx.x == 0) {
        double CValue = 0.0;
        for(int n=0;n<BLOCK_SIZE;n++){
                CValue += As[n][n];
        }
        C[Row] = CValue;
       // printf("30 %d = %lf\n", (Row), CValue);
       // printf("ARows: %d\n",ARows);
    }
}

__global__ void normalize(double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, int ARows, int ACols, int Blength, int CRows, int CCols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ARows || i >= Blength || i >= CRows 
        || j >= ACols || j >= CCols) return;
    
    int Cindex = (i * CCols) + j;
    int Aindex = (i * ACols) + j;
    C[Cindex] = A[Aindex] / B[i];
}

__global__ void delta_function(double* __restrict__ A, double* __restrict__ C, int ARows, int ACols, int CRows, int CCols, double val) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ARows || i >= CRows || j >= ACols || j >= CCols)
        return;
    int Cindex = (i * CCols) + j;
    int Aindex = (i * ACols) + j;
    C[Cindex] -= val * A[Aindex];
}