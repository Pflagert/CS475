///
/// vecAddKernel00.cu
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-16 DVN
///
/// This Kernel adds two Vectors A and B in C on GPU
/// with coalesced memory access.
/// 
#include <stdio.h>
#include <stdlib.h>
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    int max = blockDim.x * gridDim.x * N;
    while ( i < max ) {
        C[i] = A[i] + B[i];
        i += blockDim.x * gridDim.x;
    }
}