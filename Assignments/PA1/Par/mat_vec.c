
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

int main(int argc, char **argv) 
{
    int size, i, j, 
    N = 50, 
    M = 40, 
    max = 0;
    
    double *c;
    double time;

    if ( argc > 1 ) 
        N  = atoi(argv[1]);
    if ( argc > 2 ) 
        M  = atoi(argv[2]);

    printf("N=%d, M=%d\n", N, M);
    size = N * sizeof(double);
    c = (double *)malloc(size);
    
    c(0) = 0;
    /* Start Timer */
    initialize_timer ();
    start_timer();
    
    /* Compute */
    for ( j=1; j < M; j++ )
        c(0)+=j;
    
    max = c(0);
    
   #pragma omp parallel for
    for ( i=1 ; i < N ; i++ )
        c(i) = ( (M * i ) + max ) ;

    /* stop timer */
    stop_timer();
    time=elapsed_time ();

    /* print results */
    for ( i=0 ; i < N ; i+= N/8 )
        printf("c[%d] = %lf\n", i, c(i));

    printf("elapsed time = %lf\n", time);
    
    return 0;
}
