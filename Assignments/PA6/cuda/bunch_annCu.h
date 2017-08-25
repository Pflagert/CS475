#include "matmult/matmultKernels.h"
#include "func/functionKernels.h"
#define Y1(i,j) Y1[((i)*(A))+(j)]
#define Yf(i,j) Yf[((i)*(B1))+(j)]
#define Y2(i,j) Y2[((i)*(C))+(j)]
#define Z1(i,j) Z1[((i)*(C))+(j)]
#define X1(i,j) X1[((i)*(B))+(j)]
#define X2(i,j) X2[((i)*(C))+(j)]
#define Y(i,j) Y[((i)*(B))+(j)]
#define Z(i,j) Z[((i)*(B))+(j)]
//#define I(i,j) I[((i)*(A))+(j)]
#define foo(a,b) b?tanh(a):exp(a)

void displayMatrix3 (const char *label, double *m, int rows, int cols)
{
    printf ("\n%s:\n", label);
    for(int i = 0; i < rows; ++i )
    {
        for(int j = 0; j < cols; ++j )
            printf("%10.5lf\t",m[(i*cols)+j]);
        printf ("\n");
    }
}

void initializeW(double* X1, long A, long B)
{
    /*Initializes the weights*/
    long i,j;
    for (i=0; i<A;i++)
        for (j=0; j<B;j++)
            X1(i,j) = ((double)rand() / (double)RAND_MAX) * 0.2 - 0.1;
        
}

void initializeI(double* X1, long A, long B)
{
    /*Initializes the inputs*/
    long i,j;
    for (i=0; i<A;i++)
        for (j=0; j<B;j++)
            X1(i,j) = j%2;
        
}

void initializeO(double* X1, long A, long B)
{
    /*Initializes the outputs*/
    long i,j;
    for (i=0; i<A;i++)
        for (j=0; j<B;j++)
            X1(i,j) = i%2;
        
}

void mm_old(double* X2, double* Y, double* Z1, long A, long B, long C)
{
    /*Performs Matrix-Matrix Mulitplication*/
    long i,j,k;
    for (i=0; i<A; i++) 
        for (j=0; j<B; j++)
            for(k=0; k<C; k++) 
            {
                if(j==0) X2(i,k)=0;
                X2(i,k) += Y(i,j) * Z1(j,k);
            }
}

void mm(double* X2, double* Y, double* Z1, int XRows, int XCols, int YRows, int YCols, int ZRows, int ZCols,  double *deviceX2, double *deviceY, double *deviceZ1)
{
    int count=0;
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid;
    dimGrid.x = (XCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (XRows + dimBlock.y - 1)/dimBlock.y;
    
    size_t size = (YRows * sizeof(double) * YCols);   
    if(!deviceY) {    
        count++;
        cudaMalloc((void**) &deviceY, size);
        cudaMemcpy(deviceY, Y, size, cudaMemcpyHostToDevice);
    }
    if(!deviceZ1){
        count++;
        size = (ZRows * sizeof(double) * ZCols);
        cudaMalloc((void**) &deviceZ1, size);
        cudaMemcpy(deviceZ1, Z1, size, cudaMemcpyHostToDevice);
    }

    
    size = (XRows * sizeof(double) * XCols);
    if(!deviceX2){
        count++;
        cudaMalloc((void**) &deviceX2, size);
        cudaMemcpy(deviceX2, X2, size, cudaMemcpyHostToDevice);
    }

    
    MatMul<<<dimGrid, dimBlock>>>(deviceY, deviceZ1, deviceX2, YRows, YCols, ZRows,ZCols,XRows,XCols);
    cudaThreadSynchronize();
    
    if(count==3)
        cudaMemcpy(X2, deviceX2, size , cudaMemcpyDeviceToHost);
    if(count >= 2)
        cudaFree(deviceY);
    if(count >= 1)
        cudaFree(deviceZ1);
}

void mtm_old(double* X2, double* Y1, double* Z1, long A, long B, long C)
{
    /*Performs Transposed Matrix- Matrix Mulitplication*/
    long i,j,k;
    for (i=0; i<A; i++) 
        for (j=0; j<B; j++)
            for(k=0; k<C; k++)
            { 
                if(j==0) X2(i,k)=0;
                X2(i,k) += Y1(j,i) * Z1(j,k);
            }
}


void mtm(double* X2, double* Y1, double* Z1, int XRows, int XCols, int YRows, int YCols, int ZRows, int ZCols, double *deviceX2, double *deviceY1, double *deviceZ1)
{   
    int count =0;
    size_t size = (YRows * sizeof(double) * YCols);
    if(!deviceY1) {
        count++;
        cudaMalloc((void**) &deviceY1, size);
        cudaMemcpy(deviceY1, Y1, size, cudaMemcpyHostToDevice);
    }
    
    if(!deviceZ1) {
        count++;
        size = (ZRows * sizeof(double) * ZCols);
        cudaMalloc((void**) &deviceZ1, size);
        cudaMemcpy(deviceZ1, Z1, size, cudaMemcpyHostToDevice);
    }
    
    size = XRows * sizeof(double) * XCols;
    if(!deviceX2) {
        count++;
        cudaMalloc((void**) &deviceX2, size);
        cudaMemcpy(deviceX2, X2, size, cudaMemcpyHostToDevice);
    }
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid;
    dimGrid.x = (XCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (XRows + dimBlock.y - 1)/dimBlock.y;
    
    TMatMul<<<dimGrid, dimBlock>>>(deviceY1, deviceZ1, deviceX2, YRows, YCols, ZRows,ZCols,XRows,XCols);
    cudaThreadSynchronize();
    if(count==3)
        cudaMemcpy(X2, deviceX2, size , cudaMemcpyDeviceToHost);
    if(count >= 2)
        cudaFree(deviceY1);
    if(count >=1)
        cudaFree(deviceZ1);
}

void mmt_old(double* X1, double* Y2, double* Z1, long A, long B, long C) {
    long i,j,k;
    for (i=0; i<A; i++) 
        for (j=0; j<B; j++)
        {
            X1(i,j)=0;
            for(k=0; k<C; k++)
                X1(i,j) += Y2(j,k) * Z1(i,k);
            //printf("%d = %lf - O\n",((i)*(B))+(j) ,X1(i,j));
        }
        
        //displayMatrix3("Original: ",X1,A,B);
        //exit(0);
}

void mmt(double* X1, double* Y2, double* Z1,  int XRows, int XCols, int YRows, int YCols, int ZRows, int ZCols, double *deviceX1, double *deviceY2, double *deviceZ1)
{
    int count = 0;
    size_t size = (YRows * sizeof(double) * YCols);   
    if(!deviceY2) {
        count++;
        cudaMalloc((void**) &deviceY2, size);
        cudaMemcpy(deviceY2, Y2, size, cudaMemcpyHostToDevice);
    }
    
    if(!deviceZ1) {
        count++;
        size = (ZRows * sizeof(double) * ZCols);
        cudaMalloc((void**) &deviceZ1, size);
        cudaMemcpy(deviceZ1, Z1, size, cudaMemcpyHostToDevice);
    }
    
    size = XRows * sizeof(double) * XCols;
    if(!deviceX1) {
        count++;
        cudaMalloc((void**) &deviceX1, size);
        cudaMemcpy(deviceX1, X1, size, cudaMemcpyHostToDevice);
    }
    
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid;
    
    /* int maxCols = (XCols > YCols) ? (XCols) : (YCols);
     *    maxCols = (maxCols > ZCols) ? (maxCols) : (ZCols);
     *    int maxRows = (XRows > YRows) ? (XRows) : (YRows);
     *    maxRows = (maxRows > ZRows) ? (maxRows) : (ZRows); */
    int maxCols = XCols;
    int maxRows = XRows;
    dimGrid.x = (maxCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (maxRows + dimBlock.y - 1)/dimBlock.y;
    
    MatMulT<<<dimGrid, dimBlock>>>(deviceY2, deviceZ1, deviceX1, YRows, YCols,ZRows,ZCols,XRows,XCols);
    cudaThreadSynchronize();
    
    if(count==3)
        cudaMemcpy(X1, deviceX1, size , cudaMemcpyDeviceToHost);
    if(count >= 1) {
        cudaFree(deviceZ1);
    }
    if(count >= 2) {
        cudaFree(deviceY2);
    }
    /* displayMatrix3("Cuda",X1,XRows,XCols);
     *    exit(0); */
}


void func_old(double* X1, double* Yf, long A, long B1, long val)
{
    /*Performs a point-wise operation*/
    long B=B1+val;
    long i,j;
    for (i=0; i<A; i++) 
        for (j=0; j<B1; j++) {
            X1(i,(j+val)) = foo(Yf(i,j),val);
        }
        
        /*displayMatrix3("Original",X1,A,B);
         *    exit(0);*/
}

void func(double* X1, double* Yf, int  XRows, int XCols, int YRows, int YCols, long val, double *deviceX, double *deviceY){

    int count = 0;
    size_t size = (YRows * sizeof(double) * YCols);  
    if(!deviceY){
        count++;
        cudaMalloc((void**) &deviceY, size);
        cudaMemcpy(deviceY, Yf, size, cudaMemcpyHostToDevice);
    }

    
    size = XRows * sizeof(double) * XCols;
    if(!deviceX){
        count++;
        cudaMalloc((void**) &deviceX, size);
        cudaMemcpy(deviceX, X1, size, cudaMemcpyHostToDevice);
    }
    
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid;
    dimGrid.x = (XCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (XRows + dimBlock.y - 1)/dimBlock.y;
    
    function<<<dimGrid, dimBlock>>>(deviceY, deviceX, YRows, YCols, XRows, XCols, val);
    cudaThreadSynchronize();
    
    if(count==2)
        cudaMemcpy(X1, deviceX, size , cudaMemcpyDeviceToHost);
    if(count >= 2)
        cudaFree(deviceX);
    if(count >= 1)
        cudaFree(deviceY);
    
    /* displayMatrix3("Cuda",X1,XRows,XCols);
     *    exit(0); */
}

void gradient_func_old(double* X1, double* Yf, long A, long B)
{
    /*Performs a point-wise operation*/
    long B1=B+1;
    long i,j;
    for (i=0; i<A; i++)
        for (j=0; j<B; j++)  {
            X1(i,j) = Yf(i, (j+1))*(1 - pow (tanh (X1(i,j)), 2));
            //printf("%d = A[%d] * (1 - pow( tanh(C[%d]),2))\n", (((i)*(B))+(j)) , (((i)*(B1))+(j+1)), (((i)*(B))+(j)));
        }
        
        //displayMatrix3("Original",X1,A,B);
        //exit(0);
}

void gradient_func(double* X1, double* Yf, int  XRows, int XCols, int YRows, int YCols, double *deviceX, double *deviceY){
    
    int count = 0;
    size_t size = (YRows * sizeof(double) * YCols);
    if(!deviceY){
        count++;
        cudaMalloc((void**) &deviceY, size);
        cudaMemcpy(deviceY, Yf, size, cudaMemcpyHostToDevice);
    }

    
    size = XRows * sizeof(double) * XCols;
    if(!deviceX){
        count++;
        cudaMalloc((void**) &deviceX, size);
        cudaMemcpy(deviceX, X1, size, cudaMemcpyHostToDevice);
    }

    
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid;
    dimGrid.x = (XCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (XRows + dimBlock.y - 1)/dimBlock.y;
    
    gradient_function<<<dimGrid, dimBlock>>>(deviceY, deviceX, YRows, YCols, XRows, XCols);
    cudaThreadSynchronize();
    
    cudaMemcpy(X1, deviceX, size , cudaMemcpyDeviceToHost);
    if(count >= 2)
        cudaFree(deviceX);
    if(count >= 1)
        cudaFree(deviceY);
}


void error_old(double* X1, double* Y, double* Z,  long A, long B)
{
    /*Calculates the Error*/
    long i,j;
    for (i=0; i<A; i++)
        for (j=0; j<B; j++)
            X1(i,j) = Y(i,j)-Z(i,j); 
}

void error(double* X1, double* Y, double* Z,  int XRows, int XCols, int YRows, int YCols, int ZRows, int ZCols, double *deviceX1, double *deviceY)
{    
    int count=0;
    size_t size = (YRows * sizeof(double) * YCols);
    double *deviceZ;
    if(!deviceY){
        count++;
        cudaMalloc((void**) &deviceY, size);
        cudaMemcpy(deviceY, Y, size, cudaMemcpyHostToDevice);
    }

    //if(!deviceZ){
        count++;
        size = (ZRows * sizeof(double) * ZCols);
        cudaMalloc((void**) &deviceZ, size);
        cudaMemcpy(deviceZ, Z, size, cudaMemcpyHostToDevice);
    //}

    
    size = (XRows * sizeof(double) * XCols);
    if(!deviceX1){
        count++;
        cudaMalloc((void**) &deviceX1, size);
        cudaMemcpy(deviceX1, X1, size, cudaMemcpyHostToDevice);
    }
    
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid;
    dimGrid.x = (XCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (XRows + dimBlock.y - 1)/dimBlock.y;
    
    error_function<<<dimGrid, dimBlock>>>(deviceY, deviceZ, deviceX1, YRows, YCols, ZRows,ZCols,XRows,XCols);
    cudaThreadSynchronize();
    if(count == 3);
    cudaMemcpy(X1, deviceX1, size , cudaMemcpyDeviceToHost);
    if(count >= 1)
        cudaFree(deviceY);
    cudaFree(deviceZ);
}

void reduction_old(double* Y, double* X1, long A, long B)
{
    /*Performs the summation of probabilities*/
    long i,j;
    for (i=0; i<A; i++)
    {
        X1[i]=0;
        for (j=0; j<B; j++){
            X1[i] += Y(i,j);
            //printf("%d += A[%d]\n", i, ((i)*(B))+(j));
        }
        //printf("30 %d = %lf]\n",i, X1[i]);
    }
    //exit(0);
}

void reduction(double* Y, double* X1, int  Xlength, int YRows, int YCols,double *deviceX1, double *deviceY) {
    
    int count =0;
    size_t size = (YRows * sizeof(double) * YCols);
    if(!deviceY){
        count++;
        cudaMalloc((void**) &deviceY, size);
        cudaMemcpy(deviceY, Y, size, cudaMemcpyHostToDevice);
    }
    
    size = Xlength * sizeof(double);
    if(!deviceX1){
        count++;
        cudaMalloc((void**) &deviceX1, size);
        cudaMemcpy(deviceX1, X1, size, cudaMemcpyHostToDevice);
    }

    
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid;
    dimGrid.x = (YCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (YRows + dimBlock.y - 1)/dimBlock.y;
    
    reduction_function<<<dimGrid, dimBlock>>>(deviceY, deviceX1, YRows, YCols, Xlength);
    cudaThreadSynchronize();
    if(count==2)
        cudaMemcpy(X1, deviceX1, size , cudaMemcpyDeviceToHost);
    if(count >= 1)
        cudaFree(deviceX1);
    if(count >= 2)
        cudaFree(deviceY);
    //exit(0);
}

void prob_old(double* Y,double* Z, double* X1, long A, long B)
{
    /*Computes the normalized exponential*/
    long i,j;
    for (i=0; i<A; i++)
        for (j=0; j<B; j++)
            Z(i,j) = Y(i,j)/X1[i];
}

void prob(double* Y, double* Z, double* X1,  int YRows, int YCols, int ZRows, int ZCols, int Xlength, double *deviceX1, double *deviceY, double *deviceZ)
{
    int count=0;
    size_t size = (YRows * sizeof(double) * YCols);   
    if(!deviceY){
        count++;
        cudaMalloc((void**) &deviceY, size);
        cudaMemcpy(deviceY, Y, size, cudaMemcpyHostToDevice);
    }

    if(!deviceX1) {
        count++;
        size = (Xlength * sizeof(double));
        cudaMalloc((void**) &deviceX1, size);
        cudaMemcpy(deviceX1, X1, size, cudaMemcpyHostToDevice);
    }
    
    size = (ZRows * sizeof(double) * ZCols);
    if(!deviceZ) {
        count++;
        cudaMalloc((void**) &deviceZ, size);
        cudaMemcpy(deviceZ, Z, size, cudaMemcpyHostToDevice);        
    }
    
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid;
    dimGrid.x = (ZCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (ZRows + dimBlock.y - 1)/dimBlock.y;
    
    normalize<<<dimGrid, dimBlock>>>(deviceY, deviceX1, deviceZ, YRows, YCols, Xlength,ZRows,ZCols);
    cudaThreadSynchronize();
    if(count==3)
        cudaMemcpy(Z, deviceZ, size , cudaMemcpyDeviceToHost);
    if(count >=2 )
        cudaFree(deviceY);
    if(count >= 1)
        cudaFree(deviceX1);
}

void delta_old(double* Z, double* Y, long A, long B, double C)
{
    /*Updates the weight matrix*/
    long i,j;
    for (i=0; i<A; i++)
        for (j=0; j<B; j++) 
            Z(i,j) -= C*Y(i,j); 
}

void delta(double* Z, double* Y, int  ZRows, int ZCols, int YRows, int YCols, double val, double *deviceZ, double *deviceY){
    
    int count = 0;
    size_t size = (YRows * sizeof(double) * YCols);
    if(!deviceY) {
        count++;
        cudaMalloc((void**) &deviceY, size);
        cudaMemcpy(deviceY, Y, size, cudaMemcpyHostToDevice);
    }
    
    size = ZRows * sizeof(double) * ZCols;
    if(!deviceZ) {
        count++;
        cudaMalloc((void**) &deviceZ, size);
        cudaMemcpy(deviceZ, Z, size, cudaMemcpyHostToDevice);
    }

    
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid;
    dimGrid.x = (ZCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (ZRows + dimBlock.y - 1)/dimBlock.y;
    
    delta_function<<<dimGrid, dimBlock>>>(deviceY, deviceZ, YRows, YCols, ZRows, ZCols, val);
    cudaThreadSynchronize();
    if(count==2)
        cudaMemcpy(Z, deviceZ, size , cudaMemcpyDeviceToHost);
    if(count >= 2)
        cudaFree(deviceZ);
    if(count >= 1)
        cudaFree(deviceY);
}

