/*---------------------------------------------------------------------------------------------------------------*/
/// bpl.c
/// For CSU CS475 Fall 2016
/// Instructor: Sanjay Rajopadhye
/// GTA: Swetha Varadarajan
/// Based on code Created by Paul Tero at Existor Ltd as part of a neural networks tutorial
/// Modified by Swetha Varadarajan
/// Created: 2016-11-16
/*---------------------------------------------------------------------------------------------------------------*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <math.h> 

#include "timer.h"
#include "util.h"
#include "bunch_annCu.h"

#define X(i,j) X[((i)*(cmdLineArgs.N+1))+(j)]
#define H(i,j) H[((i)*(cmdLineArgs.M+1))+(j)]

/*---------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------Variable Declaration------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
/*Array description and its size in the comments next to its declation*/

double *inputs;//Given inputs = total number of samples(S)*number of inputs per sample(N) 
double *outputs;//Expected outputs = total number of samples(S)*number of outputs per sample(P) 

double *X;//Input for a given iteration = bunch size(I)*number of inputs per sample(N+1(bias))
double *device_X;
double *Y;//Output for a given iteration = bunch size(I)*number of outputs per sample(P)
double *device_Y;

double *Wxh; //Weights in between input and hidden layer = (N+1)*M
double *device_Wxh;
double *Why; //Weights in between input and hidden layer = (M+1)*P
double *device_Why;
double *dWxh; //Error Weights in between input and hidden layer = (N+1)*M
double *device_dWxh;
double *dWhy; //Error Weights in between input and hidden layer = (M+1)*P
double *device_dWhy;

double *Zh; //Weighted sum for hidden layer=I*M
double *device_Zh;
double *H;  // Activation values = I*(M+1)
double *device_H;
double *Zy; //Weighted sum for output layer=I*P
double *device_Zy;
double *E;  //Calculated Errors = I*P
double *device_E;
double *P1; //Oredicted output = I*P
double *device_P1;
double *P;  // (exp(Zy)) = I*P
double *device_P;
double *sum; //(summation of the P[i]s) = I
double *device_Sum;

double learningrate = 0.0001; /*learning rate */
long b, k2, k3;
Params cmdLineArgs;

void copyAllFromHost(int allocate) {
    size_t size = (b)*sizeof(double);
    if(allocate)
        cudaMalloc((void**) &device_Sum, size);
    cudaMemcpy(device_Sum, sum, size, cudaMemcpyHostToDevice);
    
    size = cmdLineArgs.N+1) * sizeof(double) *cmdLineArgs.M;
    if(allocate) {
        cudaMalloc((void**) &device_Wxh, size);
        cudaMalloc((void**) &device_dWxh, size);
    }
    cudaMemcpy(device_Wxh, Wxh, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dWxh, dWxh, size, cudaMemcpyHostToDevice);

    size = cmdLineArgs.M+1) * sizeof(double) *cmdLineArgs.P;
    if (allocate) {
        cudaMalloc((void**) &device_dWhy, size);
        cudaMalloc((void**) &device_Why, size);
    }
    cudaMemcpy(device_dWhy, dWhy, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Why, Why, size, cudaMemcpyHostToDevice);

    size = b*sizeof(double) * (cmdLineArgs.N+1);
    if(allocate)
        cudaMalloc((void**) &device_X, size);
    cudaMemcpy(device_X, X, size, cudaMemcpyHostToDevice);
    
    size = b*sizeof(double) * (cmdLineArgs.M+1);
    if(allocate)
        cudaMalloc((void**) &device_H, size);
    cudaMemcpy(device_H, H, size, cudaMemcpyHostToDevice);

    size = b*sizeof(double) * (cmdLineArgs.P);
    if(allocate) {
        cudaMalloc((void**) &device_E, size);
        cudaMalloc((void**) &device_P1, size);
        cudaMalloc((void**) &device_P, size);
        cudaMalloc((void**) &device_Zh, size);
        cudaMalloc((void**) &device_Zy, size);
    }
    cudaMemcpy(device_E, E, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_P, P, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_P, P1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Zh, Zh, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Zy, Zy, size, cudaMemcpyHostToDevice);
}

void copyResult() {
    size_t size = cmdLineArgs.N+1) * sizeof(double) *cmdLineArgs.M;
    cudaMemcpy(Wxh, device_Wxh, size , cudaMemcpyDeviceToHost);
    size = cmdLineArgs.M+1) * sizeof(double) *cmdLineArgs.P;
    cudaMemcpy(Why, device_Why, size , cudaMemcpyDeviceToHost);
}

void cudaFreeAll() {
    cudaFree(device_dWhy);
    cudaFree(device_dWxh);
    cudaFree(device_E);
    cudaFree(device_H);
    cudaFree(device_P);
    cudaFree(device_P1);
    cudaFree(device_Sum);
    cudaFree(device_Why);
    cudaFree(device_Wxh);
    cudaFree(device_X);
    cudaFree(device_Y);
    cudaFree(device_Zh);
    cudaFree(device_Zy);
}

void forwardPhase(long b) {
    mm(Zh,X,Wxh, b, cmdLineArgs.M, b, cmdLineArgs.N+1, cmdLineArgs.N+1, cmdLineArgs.M); //Zh=X*Wxh
    //func_old(H,Zh,b,cmdLineArgs.M,1); //H=f1(Zh)
    func(H,Zh,b,cmdLineArgs.M+1, b, cmdLineArgs.M, 1);
    
    mm(Zy,H,Why, b, cmdLineArgs.P, b, cmdLineArgs.M+1, cmdLineArgs.M+1, cmdLineArgs.P); //Zy=H*Why	
    
    //func_old(P,Zy,b,cmdLineArgs.P,0); //P=fn(Zy)	
    func(P,Zy,b,cmdLineArgs.P,b,cmdLineArgs.P,0);
    
    //reduction_old(P,sum,b,cmdLineArgs.P);
    reduction(P,sum,b,cmdLineArgs.P,b);  //summation of probabilities for each training sample
    //prob_old(P,P1,sum,b,cmdLineArgs.P); //P1=fn(P,sum)	
    prob(P,P1,sum, b, cmdLineArgs.P, b, cmdLineArgs.P, b);
    //error_old(E,P1,Y,b,cmdLineArgs.P);	//E=P1-Y
    error(E,P1,Y,b,cmdLineArgs.P,b,cmdLineArgs.P,b,cmdLineArgs.P);
}

void propagationPhase(long b) {
    //mtm_old(dWhy,H,E,cmdLineArgs.M+1,b,cmdLineArgs.P); //dWhy=H'*E ('->transpose)	
    mtm(dWhy,H,E,cmdLineArgs.M+1,cmdLineArgs.P,b,cmdLineArgs.M+1,b,cmdLineArgs.P); //dWhy=H'*E ('->transpose
    //delta_old(Why,dWhy,cmdLineArgs.M+1,cmdLineArgs.P,learningrate); //Why=fn(dwhy)                          *
    delta(Why,dWhy,cmdLineArgs.M+1,cmdLineArgs.P,cmdLineArgs.M+1,cmdLineArgs.P,learningrate); 
     //mmt_old(H,Why,E,b,cmdLineArgs.M+1,cmdLineArgs.P);
     mmt(H,Why,E,b,cmdLineArgs.M+1,cmdLineArgs.M+1,cmdLineArgs.P,b,cmdLineArgs.P); //H=Why*E'	
     //gradient_func_old(Zh,H,b,cmdLineArgs.M); //Zh=f1"(H) ("->gradient of f1)	
     gradient_func(Zh,H,b,cmdLineArgs.M,b,cmdLineArgs.M+1); //Zh=f1"(H) ("->gradient of f1)		
     
     //mtm_old(dWxh,X,Zh,cmdLineArgs.N+1,b,cmdLineArgs.M); //dWhy=H'*E ('->transpose)	
     mtm(dWxh,X,Zh,cmdLineArgs.N+1,cmdLineArgs.M,b,cmdLineArgs.N+1,b,cmdLineArgs.M);	//dWxh=X'Zh
     
     //delta_old(Wxh,dWxh,cmdLineArgs.N+1,cmdLineArgs.M,learningrate);//Wxh=fn(dWxh)
     delta(Wxh,dWxh,cmdLineArgs.N+1,cmdLineArgs.M,cmdLineArgs.N+1,cmdLineArgs.M,learningrate);//Wxh=fn(dWxh) */
}

int main(int argc, char** argv) 
{
    
    /*---------------------------------------------------------------------------------------------------------------*/
    /*-----------------------------------------Command line parsing--------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/
    

    parseCmdLineArgs(&cmdLineArgs,argc,argv);
    b = cmdLineArgs.sample_per_iter;    
    k2 = cmdLineArgs.sample_total/b ; /*number of full bunches */
    k3 = cmdLineArgs.sample_total-(k2*b); /* size of the partial bunch */
    
    /*---------------------------------------------------------------------------------------------------------------*/
    /*-------------------------------------------Memory allocations--------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/
    
    inputs  = (double*)malloc(cmdLineArgs.sample_total * sizeof(double) * cmdLineArgs.N);
    outputs = (double*)malloc(cmdLineArgs.sample_total * sizeof(double) * cmdLineArgs.P);
    
    sum	  = (double*)malloc((b)*sizeof(double));
    
    Wxh     = (double*)malloc((cmdLineArgs.N+1) * sizeof(double) *cmdLineArgs.M);
    Why	  = (double*)malloc((cmdLineArgs.M+1) * sizeof(double) *cmdLineArgs.P);
    dWxh    = (double*)malloc((cmdLineArgs.N+1) * sizeof(double) *cmdLineArgs.M);
    dWhy	  = (double*)malloc((cmdLineArgs.M+1) * sizeof(double) *cmdLineArgs.P);
    
    X	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.N+1));
    E	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
    P	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
    P1  	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
    H	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.M+1));
    Zh  	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.M));
    Zy  	  = (double*)malloc(b*sizeof(double) * (cmdLineArgs.P));
    
    if( inputs == NULL || outputs == NULL || X == NULL|| H == NULL || dWxh == NULL || dWhy == NULL 
        || Zh == NULL || Zy == NULL || Wxh == NULL || Why == NULL|| E == NULL || P == NULL
        || P1 == NULL || sum == NULL)
    {
        printf( "Could not allocate memory\n" );
        exit(0);
    }
    /*---------------------------------------------------------------------------------------------------------------*/
    /*----------------------------------------------Initializations--------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/
    
    initializeW(Wxh,(cmdLineArgs.N+1),cmdLineArgs.M);
    initializeW(Why,(cmdLineArgs.M+1),cmdLineArgs.P);
    initializeI(inputs,cmdLineArgs.sample_total,cmdLineArgs.N);
    initializeO(outputs,cmdLineArgs.sample_total,cmdLineArgs.P);
    copyAllFromHost(1);
    
    /*---------------------------------------------------------------------------------------------------------------*/
    /*------------------------------------------------Training-------------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/
    initialize_timer();
    start_timer();
    
    for (long t=0; t<cmdLineArgs.iter; t++) //Time loop
    {
        for (long s=0; s<k2; s++) //Bunch loop
        { 	
            for(long i=0;i<b;i++)
            {
                X(i,0)=H(i,0)=1;//bias setting
                //required input/output are copied from inputs/outputs to X and Y
                memcpy (&X(i,1), &inputs[cmdLineArgs.N*((s*b)+i)], cmdLineArgs.N*sizeof(double)); 
            }
            Y = &outputs[s*b*cmdLineArgs.P]; 
            forwardPhase(b);
            propagationPhase(b);
        }
        if(k3)
        {
            for(long i=0;i<k3;i++)
            {
                X(i,0)=H(i,0)=1;
                memcpy (&X(i,1), &inputs[cmdLineArgs.N*((k2*b)+i)], cmdLineArgs.N*sizeof(double));
            }
            Y = &outputs[k2*b*cmdLineArgs.P];
            forwardPhase(k3);
            propagationPhase(k3);            
        }	
    }
    
    stop_timer();
    double time = elapsed_time();
    printf( "Time: %lf\n",time);
    copyResult();
    cudaFreeAll();
    /*---------------------------------------------------------------------------------------------------------------*/
    /*----------------------------------------------Print outputs----------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/
    if(cmdLineArgs.V)
    {
        /*Need the following 2 statements for Testing*/
        displayMatrix1 ("input/hidden weights", Wxh, cmdLineArgs.N+1, cmdLineArgs.M);
        displayMatrix1 ("hidden/output weights", Why, cmdLineArgs.M+1, cmdLineArgs.P);
        /* Useful for analyzing the accuracy of prediction */
        /*if(k3){	
            displayVector ("last input", &X[k3-1], cmdLineArgs.N);
            displayVector ("last output", &Y[k3-1], cmdLineArgs.P);
            displayVector ("predicted output",&P1[k3-1], cmdLineArgs.P);
        }
        else
        {
            displayVector ("last input", &X[b-1], cmdLineArgs.N);
            displayVector ("last output", &Y[b-1], cmdLineArgs.P);
            displayVector ("predicted output",&P1[b-1], cmdLineArgs.P);
        }
        */
    }
    /*---------------------------------------------------------------------------------------------------------------*/
    /*----------------------------------------------Free Memory------------------------------------------------------*/
    /*---------------------------------------------------------------------------------------------------------------*/
    free(inputs);
    free(outputs);
    free(X);
    free(Zh);
    free(Zy);
    free(H);
    free(E);
    free(P);
    free(P1);
    free(sum);
    free(Wxh);
    free(Why);
    free(dWxh);
    free(dWhy);
    /*-------------------------------------------------------END-----------------------------------------------------*/
    return 0;
}

