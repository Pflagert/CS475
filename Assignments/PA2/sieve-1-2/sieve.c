/*/////////////////////////////////////////////////////////////////////////////
//
// File name : sieve.c
// Author    : Nissa Osheim
// Date      : 2010/19/10
// Desc      : Finds the primes up to N
//
// updated Wim Bohm
/////////////////////////////////////////////////////////////////////////////*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

#define mark(index)     mark[ ( (index-1)/2 ) + 1 ]
 
int main(int argc, char **argv) {

    char *mark;

    long size, k, i, j, n, count, N  = 100;

    /* Time */

    double time;

    if ( argc > 1 ) N  = atoi(argv[1]);

    /* Start Timer */

    initialize_timer ();
    start_timer();

    size = ( (N*sizeof(char)) / 2) + 1;
    mark = (char *)malloc(size);
    
    mark(2) = 0;
    #pragma omp parallel for
    for (i=3; i<=N; i+=2){
        mark(i) =  0;
    }

    k = 3;

    while ( (k * k) <=N ) {
        #pragma omp parallel for
        for (i=(k*k); i<=N; i+=(2*k)){
            mark(i)=1;
        }
        do k+=2; while(mark(k));
    }
    
    /* stop timer */
    stop_timer();
    time=elapsed_time ();

    /*number of primes*/
    count = 1;
    for(i = 3; i <=N; i+=2){
            if(mark(i) == 0) {
                    //printf("\t prime %ld  \n",i );
                    ++count;
            }
    }
    printf("There are %ld primes less than or equal to %ld\n", count, N);
    /* print results */
    printf("First three primes:");
    j = 1;
    printf("%d ", 2);
    for ( i=3 ; i <= N && j < 3; i+=2 ) {
        if (mark(i)==0){
                printf("%ld ", i);
                ++j;
        }
    }
    printf("\n");

    printf("Last three primes:");
    j = 0;
    n=(N%2?N:N-1);
    for (i = n; i > 1 && j < 3; i-=2){
        if (mark(i)==0){
            printf("%ld ", i);
            j++;
        }
    }
    printf("\n");


    printf("elapsed time = %lf (sec)\n", time);

    free(mark);
    return 0;
}


