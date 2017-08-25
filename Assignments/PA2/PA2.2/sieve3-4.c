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

#define MIN(x, y)             (((x) < (y)) ? (x) : (y))
#define mark(index)         mark[ ( (index-1)/2 ) + 1 ]
 
int main(int argc, char **argv) {

    char *mark;
    long *primes;

    long size, k, i, j, n, count, index, start, prime, min, blockStart, sqrtN,
    N  = 100, blockSize = 30, numprimes = 0;
    
    /* Time */    
    double time;

    if ( argc > 1 ) N  = atoi(argv[1]);
    if ( argc > 2 ) blockSize  = atoi(argv[2]);
    sqrtN = sqrt(N);

    initialize_timer ();
    start_timer();

    size = ( ( (N + 1 )*sizeof(char)) / 2);
    mark = (char *)malloc(size);
    size = ( (sqrtN + 1)*sizeof(long) );
    primes = (long *)malloc(size);

    #pragma omp parallel for
    for(i=3;i<=N;i+=2){
        mark(i) = 0;        
    }
    
    k = primes[numprimes] = 3;
    
    while(k < sqrtN){
        for(i = (k*k); i<=sqrtN; i+=(2*k)){
            mark(i) = 1;
        }
        do k+=2; while(mark(k));
        primes[++numprimes] = k;
    }
    
    #pragma omp parallel for private(index, start, prime, min, i) schedule(static, blockSize)
    for (blockStart = k; blockStart<=N; blockStart+=blockSize){
        for(index = 0, prime = primes[index]; index <= numprimes; index++, prime = primes[index]){
            start = prime*prime;
            if( start > (blockStart+blockSize)){
                    break;
            }
            else if (start < blockStart){
                start = (prime * (blockStart / prime));
                if (start < blockStart)
                    start += ( (start + prime) < blockStart ) ? (prime * (blockStart%prime)) : prime;
            }
            if( !(start % 2) ) start += prime;
            min = MIN((start + blockSize),N);
            for(i = start; i < min; i+=(2*prime)){
                mark(i) = 1;
            }
        }
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


