/*/////////////////////////////////////////////////////////////////////////////
//
// File name  : prog1.c
// Author     : Swetha Varadarajan
// Description:  wavefront parallelization for openmp.
/////////////////////////////////////////////////////////////////////////////*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include <time.h>
#include <sys/time.h>
#include <sys/errno.h>

#define MAX(x,y)   ((x)>(y) ? (x) : (y) )
#define MIN(x,y)   ((x)>(y) ? (y) : (x) )
#define A(i,j)	   A[((i)*(N))+(j)]
#define TMAX	   10
#define MOD(x,y)   ((x)%(y))

long foo(long a, long b, long c, long d, long e){
return (a+b+c+d+e)/5 ;
}

int main(int argc, char **argv) {

   long N=100; 
   long i,j,k,p,t;
   long* A;
   int verbose=0;
   double total_time;

   if ( argc > 1 ) N  = atoi(argv[1]);
   if ( argc > 2 ) verbose=1;
   printf("N=%ld\n", N);

   A = (long *)malloc(N*N*sizeof(long));
  
   if (NULL == A) 
   {
	fprintf(stderr, "malloc failed\n");
	return(-1);
   }
   
  
   /* Initialization */
   srand((unsigned)time(NULL));
   for (i=0;i<N;i++)
   	for  (j=0;j<N;j++)
 		A(i,j)=MOD(rand(),100);



   /* Start Timer */

   initialize_timer ();
   start_timer();

   #ifdef SEQ
   /*Note: It looks like Jacobi-2D. But, it is the same array. There is no current and previous.
	Some dependencies are from current iteration of k loop whereas others are from (k-1)th iteration */

   for(k=1;k<TMAX;k++)
	   for (i=1;i<N-1;i++)
	  	for  (j=1;j<N-1;j++)
			A(i,j)=foo(A(i,j-1),A(i-1,j),A(i,j+1),A(i+1,j),A(i,j));
   #endif

   #ifdef PAR
   /*Re-write the above loop body by applying a legal transformation 
	that achieves wavefront-parallelization*/
   
   #endif

   /* stop timer */
   stop_timer();
   total_time=elapsed_time ();

   if(verbose){
	   for (i=0;i<N;i++){
		for  (j=0;j<N;j++)
			printf("A(%ld,%ld)=%ld\t",i,j,A(i,j));
		printf("\n");
	   }
   }
	
   printf("elapsed time = %lf\n", total_time);
   free(A);
   return 0;
}
