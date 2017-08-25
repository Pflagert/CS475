/// matmultKernel.h
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-19 DVN
///
/// Kernels defined with this header must 
/// multiply two matrices using CUDA: A x B = C
///

#ifndef __MMKERNEL__
#define __MMKERNEL__

// Defines for size of thread block and data computed by a thread block
#define BLOCK_SIZE 16
#define FOOTPRINT_SIZE 16

// The type Matrix is really a MATRIX DESCRIPTOR. 
// Matrices are stored in row major order:
//       M[row,col] = *(M.elements + row * M.stride + col)
//
// A sub matrix is not copied but allocated in the full matrix.
//
// This requires the stride of the full matrix to properly get to the
// next row of the sub matrix (a block).
//
// Stride is the width in bytes from one element of the larger matrix 
// to the element in the same column but one row down.


// C = A * B
__global__ void MatMul(double*__restrict__  A, double*__restrict__  B, double*__restrict__  C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols);
// C = Transpose(A) * B
__global__ void TMatMul(double*__restrict__  A, double*__restrict__  B, double*__restrict__  C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols);
// C = A * Transpose(B)
__global__ void MatMulT(double*__restrict__  A, double*__restrict__  B, double*__restrict__  C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols);
#endif

