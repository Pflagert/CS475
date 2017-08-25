#include <stdio.h>

__global__ void device_hello(){

  //uncomment this line to print only one time (unless you have multiple blocks)
  //if(threadIdx.x==0)
  int block = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y * blockDim.z) ;
  int thread = (threadIdx.z * (blockDim.x * blockDim.y) ) + (threadIdx.y * blockDim.x) + threadIdx.x;
  int global_id = block + thread;
  printf("Hello world! from the device! Global_Thread_Id:%d\n",global_id);
  return;
}

int main(void){

  // rather than calling fflush    
  setbuf(stdout, NULL);

  // greet from the host
  printf("Hello world! from the host!\n");

  // launch a kernel with a block of threads to greet from the device
  dim3 blockSize(2,2,2);
  dim3 gridSize(2,2,1);
  // run several variations by playing with the block and grid sizes
  // above -- if you change the y or z dimensions change the print 
  // statement to reflect that.
  device_hello<<<gridSize,blockSize>>>();

  // comment this line out and see what happens
  cudaDeviceSynchronize();

  printf("Goodbye world! from the host!\n");

  return 0;
}
