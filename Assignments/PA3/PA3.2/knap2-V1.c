//
// File name   : knap.c
// Author      : DaeGon
// Modified by : Sanjay and Wim 
// Date        : Sep 20
// Description : Dynamic Programming for the 0/1 knapsack
//               problem.  The entire table is saved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "timer.h"

#define    MAX(x,y)   ((x)>(y) ? (x) : (y))

int max_index(int start, int end, long *array) {
    int max = start;
    for(start=start+1; start<=end; start++){
        if( array[max] < array[start] ){
            max = start;
        }
    }
    return max;
}

long* get_last_row(long start, long end, long capacity, long *weights, long *profits) {
    long i,j,size;
    long *cur;
    size = sizeof(long)* (capacity+1);    
    cur = (long *)memset(malloc(size), 0, size);
    
    for(i=start; i<=end; i++){
        for(j=capacity;j>=0;j--){
            cur[j] = (j < weights[i]) ? cur[j] : MAX(cur[j], (cur[j-weights[i]] + profits[i]) );
        }
    }
    return cur;
}

long solve_kp_base(long i, long j, long C, long *weights, long *profits, int *solutions) {
    long profit = 0;
    if (C <= 0){
      return -1;  
    }
    else if (j<=i){
        if ( weights[i] <= C ) {
            solutions[i] = 1;
            return profits[i];
        }
        return -1;
    }
    else if( (j - i) == 1) {
        if(weights[i] + weights[j] <= C) {
            solutions[i] = 1;
            solutions[j] = 1;
            return profits[i] + profits[j];
        }
        else if (weights[i] <= C && weights[j] <= C){
                if (profits[i] > profits[j]) {
                    solutions[i] = 1;
                    return profits[i];
                }
                else {
                    solutions[j] = 1;
                    return profits[j];
                }
        }
        else if(weights[i] <= C) {
            solutions[i] = 1;
            return profits[i];
        }
        else if(weights[j] <= C) {
            solutions[j] = 1;
            return profits[j];
        }        
        return -1;
    }
    else if( (j - i) == 2) {
        if(weights[i] + weights[i+1] + weights[j] <= C) {
            solutions[i] = 1;
            solutions[i+1] = 1;
            solutions[j] = 1;
            return profits[i] + profits[i+1] + profits[j];
        }
        else {
          long temp[3];
          temp[0] = (weights[i] <= C) ? (profits[i]) : -1;
          temp[1] = (weights[i+1] <= C) ? (profits[i+1]) : -1;
          temp[2] = (weights[j] <= C) ? (profits[j]) : -1;
          int max = max_index(0,2,temp);
          if(temp[max] >= 0) {
                solutions[i+max] = 1;
                profit += profits[i+max];
                C = C-weights[i+max];
                if(max == 0){
                    temp[0] = -11;
                    temp[1] = (weights[i+1] <= C) ? temp[1] : -1;
                    temp[2] = (weights[j] <= C) ? temp[2] : -1;
                    max = max_index(0,2,temp);
                }
                else if(max == 1){
                    temp[0] = (weights[i] <= C) ? temp[0] : -1;
                    temp[1] = -11;
                    temp[2] = (weights[j] <= C) ? temp[2] : -1;
                    max = max_index(0,2,temp);
                }
                else if(max == 2){
                    temp[0] = (weights[i] <= C) ? temp[0] : -1;
                    temp[1] = (weights[i+1] <= C) ? temp[1] : -1;
                    temp[2] = -11;
                    max = max_index(0,2,temp);
                }
                if(temp[max] >= 0) {
                    solutions[i+max] = 1;
                    profit += profits[i+max];
                }
                return profit;
          }
        }
        return -1;
    }
    return 0;
}

long solve_kp(long i, long j, long C, long *weights, long *profits, int *solutions, int depth) {
        long k, cap, max, sec1, sec2;
        long *A1, *A2;
        long m = ((j-i)/2) - (j%2);
        A1 = get_last_row(i, i+m, C, weights, profits);
        A2 = get_last_row(i+m+1, j, C, weights, profits);


        cap = 0;
        max = A1[0] + A2[C-0];
        for(k=1; k<=C; k++) {
            if ( (A1[k] + A2[C-k]) > max) {
                max = A1[k] + A2[C-k];
                cap = k;
            }
        }
        
        sec1 = sec2 = max;
        
        free(A1);
        free(A2);
        
        if (sec1 = solve_kp_base(i, i+m, cap, weights, profits, solutions) == 0)
            #pragma omp task final(depth <= 1)
                sec1 =  solve_kp(i, i+m, cap, weights, profits, solutions, depth-1);
        sec1 = MAX(max,sec1);
        
        if(sec2 = solve_kp_base(i+m+1, j, C-cap, weights, profits, solutions) == 0)
            #pragma omp task final(depth <= 2)
            sec2 = solve_kp(i+m+1, j, C-cap, weights, profits, solutions, depth-2);
        sec2 = MAX(max,sec2);

        #pragma omp taskwait
        return MAX(sec1,sec2);
}

int main(int argc, char **argv) {
    FILE *fp;
    long N, C;                   // # of objects, capacity 
    long *weights, *profits;     // weights and profits
    int verbose, depth;
    // Temp variables
    long i, j, size;

    // Time
    double time;

    // Read input file (# of objects, capacity, (per line) weight and profit )
    if ( argc > 1 ) {
        fp = fopen(argv[1], "r"); 
        if ( fp == NULL) {
            printf("[ERROR] : Failed to read file named '%s'.\n", argv[1]);
            exit(1);
        }
    } 
    else {
        printf("USAGE : %s [filename].\n", argv[0]);
        exit(1);
    }

    if (argc > 2)
        depth = atoi(argv[2]);
    else
        depth = omp_get_max_threads();
    if(depth == 0)
        omp_set_nested(0);
    else
        omp_set_nested(1);
    if (argc > 3) 
        verbose = atoi(argv[3]); 
    else 
        verbose = 0;

    if (fscanf(fp, "%ld %ld", &N, &C) != 2) {
        printf("[ERROR] : Input file is not well formatted.\n");
        exit(1);
    }
    printf("The number of objects is %ld, and the capacity is %ld.\n", N, C);

    size    = N * sizeof(long);
    weights = (long *)malloc(size);
    profits = (long *)malloc(size);
    int *solution = (int *)malloc(size);

    if ( weights == NULL || profits == NULL ) {
        printf("[ERROR] : Failed to allocate memory for weights/profits.\n");
        exit(1);
    }

    for ( i=0 ; i < N ; i++ ) {
        if (fscanf(fp, "%ld %ld", &(weights[i]), &(profits[i])) != 2 ) {
            printf("[ERROR] : Input file is not well formatted.\n");
            exit(1);
        }
        solution[i] = 0;
    }

    fclose(fp);

    // Solve for the optimal profit
   
    initialize_timer ();
    start_timer();

  
    // End of "Solve for the optimal profit"
    long final = solve_kp(0,N-1, C, weights, profits, solution, depth);
    stop_timer();
    time = elapsed_time ();
    printf("The optimal profit is %ld \nTime taken : %lf.\n", final, time);
     

    if (verbose==1) {
        printf("Solution vector is: ");
        for (i=0 ; i<N ; i++ ) {
            printf("%d ", solution[i]);
        }
        printf("\n");
    }
    
    free(weights);
    free(profits);
    free(solution);

    return 0;
}
