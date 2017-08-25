/*/////////////////////////////////////////////////////////////////////////////
//
// File name  : prog1.c
// Author     : Swetha Varadarajan
// Description:  wavefront paralellization for openmp.
/////////////////////////////////////////////////////////////////////////////*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

#define MAX(x,y)   	((x)>(y) ? (x) : (y) )
#define MIN(x,y)   	((x)>(y) ? (y) : (x) )
#define foo(x,y)   	((x)+(y))/2
#define A(i,j)	   	A[((i)*(N))+(j)]

int main(int argc, char **argv) {

   long N=100; 
   long i,j,p,t;
   long* A;
   int verbose=0;
   double time;

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
   A(0,0)=100;
   for (i=1;i<N;i++)
   {
   	A(0,i)=foo(A(0,i-1),i);
	A(i,0)=foo(i,A(i-1,0));
   }


   /* Start Timer */

   initialize_timer ();
   start_timer();

   #ifdef SEQ
   for (i=1;i<N;i++)
  	for  (j=1;j<N;j++)
		A(i,j)=foo(A(i,j-1),A(i-1,j));
   #endif

   #ifdef PAR1
   /*Applying the transformation: (i,j->p,t) = (i,j->i,i+j-1) from lecture slides*/
   for (t=1; t<=N+N-3; t++)
	#pragma omp parallel for private(i,j,p) schedule(dynamic)
  	for (p=MAX(1,t-N+2); p<=MIN(t, N-1); p++){
		i=p;
		j=t-p+1;
	        A(i,j)=foo(A(i,j-1),A(i-1,j));
	}
    #endif

    #ifdef PAR2
   /*Applying the transformation: (i,j->p,t) = (i,j->j,i+j-1)
	Work-out the loop bounds for this transformation using 
	the concepts from the lecture. */
   for (t=1; t<=N+N-1; t++)
       #pragma omp parallel for private(i,j,p) schedule(dynamic)
       for (p=MAX(t-N-2,1); p<=MIN(N-1, t-1); p++){
           i=p;
           j=t-p;
           A(i,j)=foo(A(i,j-1),A(i-1,j));
       }


   #endif
   /* stop timer */
   stop_timer();
   time=elapsed_time ();

   if(verbose){
	   for (i=0;i<N;i++){
		for  (j=0;j<N;j++)
			printf("A(%ld,%ld)=%ld\t",i,j,A(i,j));
		printf("\n");
	   }
   }
	
   printf("elapsed time = %lf\n", time);
   free(A);
   return 0;
}
