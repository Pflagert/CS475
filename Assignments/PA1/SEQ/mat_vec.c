
/*/////////////////////////////////////////////////////////////////////////////
//
// File name : matvec.c
// Author: Nissa O updated: Wim B
//
/////////////////////////////////////////////////////////////////////////////*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

#define A(i,j)     A[(i)*M+j]
#define b(i)       b[i]
#define c(i)       c[i]

int main(int argc, char **argv) {

   int N  = 50;
   int M  = 40;

   double *A, *b, *c;

   int   size;
   int   i, j;

   /* Time */

   double time;

   if ( argc > 1 ) N  = atoi(argv[1]);
   if ( argc > 2 ) M  = atoi(argv[2]);

   printf("N=%d, M=%d\n", N, M);

   size = N * M * sizeof(double);
   A    = (double *)malloc(size);
   size = N * sizeof(double);
   c    = (double *)malloc(size);
   size = M * sizeof(double);
   b    = (double *)malloc(size);

   /* Initialize */
   for ( i=0 ; i < N ; i++ ) {
      for ( j=0 ; j < M ; j++ ) {
         A(i,j) = i + j;
         b(j)   = 1;
      }
   }

  /* Start Timer */
   initialize_timer ();
   start_timer();

   /* Compute */
   for ( i=0 ; i < N ; i++ ) {
      c(i) = 0;
      for ( j=0; j < M ; j++ ) {
         c(i) += A(i,j) * b(j);
      }
   }

   /* stop timer */
   stop_timer();
   time=elapsed_time ();

   /* print results */
   for ( i=0 ; i < N ; i+= N/8 ) {
      printf("c[%d] = %lf\n", i, c(i));
   }

   printf("elapsed time = %lf\n", time);
   return 0;
}
