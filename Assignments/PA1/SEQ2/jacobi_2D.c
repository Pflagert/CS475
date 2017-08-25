    /*
    * jacobi_2D.c
    *
    * Author: Rajan updated: Wim
    */
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include <math.h>
#define   INIT_VALUE       10E5
#define   prev(i,j)        prev[(i)*N+(j)]
#define   cur(i,j)         cur[(i)*N+(j)]


void printMatrix(double *data, int size);

int main(int argc, char **argv)
{

    int     N;
    int     t;
    int     MAX_ITERATION = 2000;
    double  *prev, *cur;
    double  error = INIT_VALUE;

    // Timer
    double  time;

    // temporary variables
    int     i,j;
    double  *temp;

    // Check commandline args.
    if ( argc > 1 ) {
        N = atoi(argv[1]);
    } else {
        printf("Usage : %s [N]\n", argv[0]);
        exit(1);
    }
    if ( argc > 2 ) {
        MAX_ITERATION = atoi(argv[2]);
    }

    // Memory allocation for data array.
    prev  = (double *) malloc( sizeof(double) * N * N );
    cur   = (double *) malloc( sizeof(double) * N * N );
    if ( prev == NULL || cur == NULL ) {
        printf("[ERROR] : Fail to allocate memory.\n");
        exit(1);
    }

    // Initialization
    for ( i=1 ; i < N-1 ; i++ ) {
        for ( j=1 ; j < N-1 ; j++ ) {
            prev(i,j) = 0.0;
        }
    }
    for ( i=0 ; i < N ; i++ ) {
        prev(i  , 0  )  = INIT_VALUE;
        prev(i  , N-1)  = INIT_VALUE;
        prev(0  , i  )  = INIT_VALUE;
        prev(N-1, i  )  = INIT_VALUE;
        cur( i  , 0  )  = INIT_VALUE;
        cur( i  , N-1)  = INIT_VALUE;
        cur( 0  , i  )  = INIT_VALUE;
        cur( N-1, i  )  = INIT_VALUE;
    }
    initialize_timer();
    start_timer();

    // Computation
    t = 0;


    while ( t < MAX_ITERATION) {

        // Computation
        for ( i=1 ; i < N-1 ; i++ ) {
            for ( j=1 ; j < N-1 ; j++ ) {
                cur(i,j) = ((prev(i-1,j-1)+prev(i-1,j)+prev(i-1,j+1)+prev(i,j-1)+prev(i,j)+prev(i,j+1)+prev(i+1,j-1)+prev(i+1,j)+prev(i+1,j+1)))/9;
            }
        }
        temp = prev;
        prev = cur;
        cur  = temp;
        t++;
    }

    stop_timer();
    time = elapsed_time();

    printf("Data : %d by %d , Iterations : %d , Time : %lf sec\n", N, N, t, time);
    printf("Final data\n");
    printMatrix(prev, N);



}

void printMatrix(double *data, int size) 
{
    int i,j;

    /* print a portion of the matrix */
    for ( i= size/10 ; i < size/2 ; i+=size/10 ) {
        for ( j=size/10 ; j < size/2 ; j+=size/10 ) {
            printf("%lf ", data[i*size+j]);
        }
        printf("\n");
    }

    return;
}

