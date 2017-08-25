/******************************************************************************
 * Jacobi2D benchmark
 * Basic parallelisation with OpenMP
 *
 * Usage:
 * make omp
 * aprun -n8 bin/Jacobi2D-BlockParallel-MPI \
 *     `cat src/Jacobi2D-BlockParallel-MPI.perfexecopts`
 * For a run on 8 processes
 ******************************************************************************/
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <getopt.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>


#define STENCIL(read,write,y,x) space[write][y][x] = \
                                                     ( space[read][y-1][x] +\
                                                       space[read][y][x] +\
                                                       space[read][y+1][x] +\
                                                       space[read][y][x+1] +\
                                                       space[read][y][x-1] )/5;


#include "util.h"

void packColToVec(double**data,int col,double*vec, int row_count){
  /* You are responsible for this code */
  return;
}
void unpackVecToCol(double**data,int col,double*vec, int row_count){
  /* You are responsible for this code */
  return;
}

// main
// Stages
// 1 - command line parsing
// 2 - data allocation and initialization
// 3 - jacobi 2D timed within an openmp loop
// 4 - output and optional verification
int main( int argc, char* argv[] ){

  // get started with MPI
  int p_count,rank;
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &p_count );
  MPI_Status s; 

  // rather than calling fflush    
  setbuf(stdout, NULL);

  // 1 - command line parsing
  Params cmdLineArgs;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);

  // the command line parsing as written will set the tile
  // sizes, however, for this implementation we are going to ignore that
  // and determine that each processor gets the a single sized tile
  // it is up to the person running the code to determine the size
  // and shape of the tile that they want to use.
  // However, we want to make sure that they have chosen a size
  // this code can work with
  if(cmdLineArgs.tile_len_x*cmdLineArgs.tile_len_y*p_count < 
     cmdLineArgs.problemSize*cmdLineArgs.problemSize ){
    if(rank == 0){
      fprintf(stderr,"The tile size is too small to accomidate the problem\n");
    }
    MPI_Finalize();
    return;
  }

  // 2 - data allocation and initialization
  // each PE should allocate just the tile that it is working on
  // you need:
  // 1: a mapping from PE to tile
  // 2: the tile size (cmdLineArgs.tile_len_x, cmdLineArgs.tile_len_y)
  // 3: the full data size (cmdLineArgs.problemSize the problem size is square)
  // 4: how many tiles in the x direction?
  int x_tile_count = ceild(cmdLineArgs.problemSize,cmdLineArgs.tile_len_x);
  // 5: how many tiles in the y direction?
  int y_tile_count = ceild(cmdLineArgs.problemSize,cmdLineArgs.tile_len_y);
  // Now you have all of the information 2-5 needed to determin the
  // mapping (1) Draw a picture -- number each tile -- convert 1D numbers
  // to 2D coordinates
  int tile_x = rank % x_tile_count;
  int tile_y = rank / x_tile_count;


  // the local (pe specific lower bound for the data indexes)
  int lowerBound_x = 1;
  int lowerBound_y = 1;
  int  upperBound_x = lowerBound_x + cmdLineArgs.tile_len_x;
  int  upperBound_y = lowerBound_y + cmdLineArgs.tile_len_y;

  // calculate a local tile length in case this is a partial tile
  if((tile_x*cmdLineArgs.tile_len_x)+upperBound_x>(cmdLineArgs.problemSize+1)){
    upperBound_x = cmdLineArgs.problemSize - (tile_x*cmdLineArgs.tile_len_x);
  }
  if((tile_y*cmdLineArgs.tile_len_y)+upperBound_y > cmdLineArgs.problemSize+1){
    upperBound_y = cmdLineArgs.problemSize - tile_y*cmdLineArgs.tile_len_y;
  }
  int tile_len_x = upperBound_x - lowerBound_x;
  int tile_len_y = upperBound_y - lowerBound_y;

  // Now allocate a 2D array for the PE specific data
  double** space[2];
  int i; 
  
  // I am choosing to have the x index be the most contiguous
  // so this allocates the y space
  // there are two of them because of the ping pong
  space[0] = (double**)malloc((tile_len_y + 2) * sizeof(double*));
  space[1] = (double**)malloc((tile_len_y + 2) * sizeof(double*));
  if( space[0] == NULL || space[1] == NULL ){
    printf( "Could not allocate y axis of space array\n" );
    exit(0);
  }
  
  // allocate x index space
  for( i = 0; i < tile_len_y+2; ++i ){
    space[0][i]=(double*)malloc((tile_len_x+2) * sizeof(double));
    space[1][i]=(double*)malloc((tile_len_x+2) * sizeof(double));
    if( space[0][i] == NULL || space[1][i] == NULL ){
      printf( "Could not allocate x axis of space array\n" );
      exit(0);
    }
  }
          
  // use global seed to seed the random number gen (will be constant)
  // every tile needs to do this -- but you don't want them having
  // the same see
  srand(cmdLineArgs.globalSeed+rank);
  int x, y;
  // seed the space.
  for( y = lowerBound_y; y < upperBound_y; ++y ){
    for( x = lowerBound_x; x < upperBound_x; ++x ){
      space[0][y][x] = rand() / (double)rand();
    }
  }
  
  // set outside constant values (sanity)
  // note that only the tiles that fall on the exterior
  // need to do this
  if(tile_x == 0 ){
    for( i = 0; i <= upperBound_y; ++i){
      space[0][i][0] = 0;
      space[1][i][0] = 0;
    }
  }
  if(tile_y == 0 ){
    for( i = 0; i <= upperBound_x; ++i){
      space[0][0][i] = 0;
      space[1][0][i] = 0;
    }
  }
  if(tile_x == (x_tile_count-1)){
    for( i = 0; i <= upperBound_y; ++i){
      space[0][i][upperBound_x] = 0;
      space[1][i][upperBound_x] = 0;
    }
  }
  if(tile_y == (y_tile_count-1)){
    for( i = 0; i <= upperBound_x; ++i){
      space[0][upperBound_y][i] = 0;
      space[1][upperBound_y][i] = 0;
    }
  }

  //Each PE is also going to need a buffer for sending
  //column data around allocate that here
  //double* buffer = (double*)malloc((tile_len_y+2)*sizeof(double)); 

  // 3 - jacobi 2D timed within a  loop
  // This code works for an interior tile
  double start_time = MPI_Wtime();
  int t,read=0,write=1;
  int x_lb = 1;
  int y_lb = 1; 
  int x_ub = upperBound_x;
  int y_ub = upperBound_y;

  // Initialize Variables that you need for exchange here


  for( t = 0; t < cmdLineArgs.T; ++t ){ 

    // Here is where your exchange code belongs

    /*// you may find this useful for debugging
    for( y = lowerBound_y-1; y <= upperBound_y; ++y ){
      for( x = lowerBound_x-1; x <= upperBound_x; ++x ){
        fprintf(stderr,"%d:(%d,%d) = %f\n",rank,y,x,space[0][y][x]);
      }
    }*/

    for( y = lowerBound_y; y < upperBound_y; ++y ){
      for( x = lowerBound_x; x < upperBound_x; ++x ){
        STENCIL( read, write, y, x);
      }    
    }

    read = write;
    write = 1 - write;


  /*// you may find this useful for debugging
  for( y = lowerBound_y-1; y <= upperBound_y; ++y ){
    for( x = lowerBound_x-1; x <= upperBound_x; ++x ){
      fprintf(stderr,"%d:(%d,%d) = %f\n",rank,y,x,space[read][y][x]);
    }
  }*/

    
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();
  double time =  (end_time - start_time);


  // DO NOT EDIT CODE BELOW THIS LINE!!
  // 4 - output and optional verification
  if( cmdLineArgs.printtime && rank == 0 ){
    printf( "Time: %f\n", time );
  }

  if( cmdLineArgs.verify ){

    if(rank == 0){ 
      double** test_space;
      // allocate enough space for the entire problem size
      test_space = (double**)malloc((cmdLineArgs.problemSize+2) 
                       * sizeof(double*));
      if( test_space == NULL ){
        printf( "Could not allocate y axis of space array\n" );
        exit(0);
      }
  
      // allocate x index space
      for( i = 0; i < cmdLineArgs.problemSize+2; ++i ){
        test_space[i]=(double*)malloc((cmdLineArgs.problemSize+2) 
                         * sizeof(double));
        if( test_space[i] == NULL ){
          printf( "Could not allocate x axis of test_space array\n" );
          exit(0);
        }
      }

      int j;
      for( i=1; i<=tile_len_y;i++){
        for( j=1; j<=tile_len_x;j++){
          test_space[i][j] = space[read][i][j];
        }
      }

      int pid;
      for(pid = 1; pid<p_count; pid++){
        tile_x = pid % x_tile_count;
        tile_y = pid / x_tile_count;

        // the local (pe specific lower bound for the data indexes)
        lowerBound_x = tile_x*cmdLineArgs.tile_len_x;
        lowerBound_y = tile_y*cmdLineArgs.tile_len_y;
        upperBound_x = lowerBound_x + cmdLineArgs.tile_len_x;
        upperBound_y = lowerBound_y + cmdLineArgs.tile_len_y;

        // calculate a local tile length in case this is a partial tile
        if(upperBound_x > cmdLineArgs.problemSize){
          upperBound_x = cmdLineArgs.problemSize;
        }
        if(upperBound_y > cmdLineArgs.problemSize){
          upperBound_y = cmdLineArgs.problemSize;
        }
        tile_len_x = upperBound_x - lowerBound_x;
        tile_len_y = upperBound_y - lowerBound_y;

        int y_pos;
        int x_pos = tile_x*tile_len_x + 1;
        for( i=1; i<=tile_len_y;i++){
          y_pos = lowerBound_y + i;
          MPI_Recv(&test_space[y_pos][x_pos],tile_len_x,
                   MPI_DOUBLE,pid,i,MPI_COMM_WORLD,&s);
        }
      } 
      if(!verifyResultJacobi2DTiled(test_space,
          cmdLineArgs.problemSize,
          cmdLineArgs.globalSeed,
          cmdLineArgs.T,
          x_tile_count,
          y_tile_count)){
        fprintf(stderr,"FAILURE\n");
      }else{
        fprintf(stderr,"SUCCESS\n");
      }
   // all of the other processors need to send
    }else{
      for( i=1; i<=tile_len_y;i++){
        MPI_Send(&space[cmdLineArgs.T&1][i][1],tile_len_x,MPI_DOUBLE,
                 0,i,MPI_COMM_WORLD);
      }
    }
  }

  MPI_Finalize();

  
}
