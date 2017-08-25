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

#include "matmultKernel.h"
#include <stdio.h>
#ifndef FOOTPRINT_SIZE
    #define FOOTPRINT_SIZE BLOCK_SIZE
#endif
// Define a gpu kernel to perform matrix multiplication
// of A x B = C.

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  // grid Dimensions
  int gridx = gridDim.x;
  int gridy = gridDim.y;
  
  // Iterate up to max times
  int max = (A.width / BLOCK_SIZE);
  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
    float Cvalue1, Cvalue2, Cvalue3, Cvalue4;
    Cvalue1 = Cvalue2 = Cvalue3 = Cvalue4 = 0.0f;


    // Each thread computes one element of Csub in its copy of CValue
    for (int m = 0;  m < max; ++m){
        // 4 shared matrices that will be used to compute 4 values ( A * B ) ( A * B2 ) ( A2 * B ) (A2 * B2)
        __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shared_A2[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shared_B2[BLOCK_SIZE][BLOCK_SIZE];

        // Same as MatMultKernel00.cu except I removed Asub and Bsub. Looks more complex but it is the same thing.
        shared_A[thread_row][thread_col] = A.elements[(A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m)+(thread_row * A.stride + thread_col)];
        shared_B[thread_row][thread_col] = B.elements[(B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col)+(thread_row * B.stride + thread_col)];
        
        // differs from shared_A, block_row is incremented by gridDim.y
        shared_A2[thread_row][thread_col] = A.elements[(A.stride * BLOCK_SIZE * (block_row +gridy)+ BLOCK_SIZE * m)+(thread_row * A.stride + thread_col)];
        // differs from shared_B, block_col is incremented by gridDim.x
        shared_B2[thread_row][thread_col] = B.elements[(B.stride * BLOCK_SIZE * m + BLOCK_SIZE * (block_col + gridx))+(thread_row * B.stride + thread_col)];
        // Synchronize to ensure all elements are read
        __syncthreads();

        // Do an inproduct of one row of shared_A and one col of shared_B
        #pragma unroll
        for(int e=0; e<BLOCK_SIZE; ++e) {
            Cvalue1 += shared_A[thread_row][e] * shared_B[e][thread_col];
        }
        // Do an inproduct of one row of shared_A and one col of shared_B2
        #pragma unroll
        for(int e=0; e<BLOCK_SIZE; ++e) {
            Cvalue2 += shared_A[thread_row][e] * shared_B2[e][thread_col];
        }
        // Do an inproduct of one row of shared_A2 and one col of shared_B
        #pragma unroll
        for(int e=0; e<BLOCK_SIZE; ++e) {
            Cvalue3 += shared_A2[thread_row][e] * shared_B[e][thread_col];
        }
        // Do an inproduct of one row of shared_A2 and one col of shared_B
        #pragma unroll
        for(int e=0; e<BLOCK_SIZE; ++e) {
            Cvalue4 += shared_A2[thread_row][e] * shared_B2[e][thread_col];
        }

        // Synchronize to ensure all Cvalues have been incremented
        // before reading in the next shared_A, shared_B, shared_A2, and shared_B2 BLOCKS
        __syncthreads();
    }
    // Write to GLOBAL memory.
    // Each thread writes its own cell value.
    // Write result of (A * B)
    C.elements[(C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col) + (thread_row * C.stride + thread_col)] = Cvalue1;
    // Write result of (A * B2)
    C.elements[(C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * (block_col +gridx) ) + (thread_row * C.stride + thread_col)] = Cvalue2;
    // Write result of (A2 * B)
    C.elements[(C.stride * BLOCK_SIZE * (block_row + gridy) + BLOCK_SIZE * block_col) + (thread_row * C.stride + thread_col)] = Cvalue3;
    // Write result of (A2 * B2)
    C.elements[(C.stride * BLOCK_SIZE * (block_row + gridy) + BLOCK_SIZE * (block_col +gridx)) + (thread_row * C.stride + thread_col)] = Cvalue4;
}