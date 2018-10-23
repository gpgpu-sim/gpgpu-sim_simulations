/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Walsh transforms belong to a class of generalized Fourier transformations. 
 * They have applications in various fields of electrical engineering 
 * and numeric theory. In this sample we demonstrate efficient implementation 
 * of naturally-ordered Walsh transform 
 * (also known as Walsh-Hadamard or Hadamard transform) in CUDA and its 
 * particular application to dyadic convolution computation.
 * Refer to excellent Jorg Arndt's "Algorithms for Programmers" textbook
 * http://www.jjj.de/fxt/fxtbook.pdf (Chapter 22)
 *
 * Victor Podlozhnyuk (vpodlozhnyuk@nvidia.com)
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil_inline.h>
#include <shrQATest.h>


////////////////////////////////////////////////////////////////////////////////
// Reference CPU FWT
////////////////////////////////////////////////////////////////////////////////
extern"C" void fwtCPU(float *h_Output, float *h_Input, int log2N);
extern"C" void slowWTcpu(float *h_Output, float *h_Input, int log2N);
extern "C" void dyadicConvolutionCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int log2dataN,
    int log2kernelN
);


////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
#include "fastWalshTransform_kernel.cu"



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////

// Parse program arguments
void ParseArguments(int argc, char** argv, int& logK, int& logD)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--logK") == 0 ||
            strcmp(argv[i], "-logK") == 0) 
        {
            logK = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "--logD") == 0 ||
            strcmp(argv[i], "-logD") == 0) 
        {
            logD = atoi(argv[i+1]);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]){
    float *h_Data, 
          *h_Kernel,
          *h_ResultCPU, 
          *h_ResultGPU;

    float *d_Data,
          *d_Kernel;

    double delta, ref, sum_delta2, sum_ref2, L2norm, gpuTime;

    unsigned int hTimer;
    int i;

    int log2Kernel = 7;
    int log2Data = 15;
    ParseArguments(argc,argv,log2Kernel,log2Data);

    const int   dataN = 1 << log2Data;
    const int kernelN = 1 << log2Kernel;

    const int   DATA_SIZE = dataN   * sizeof(float);
    const int KERNEL_SIZE = kernelN * sizeof(float);

    const double NOPS = 3.0 * (double)dataN * (double)log2Data / 2.0;

    shrQAStart(argc, argv);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    cutilCheckError( cutCreateTimer(&hTimer) );

    printf("Initializing data...\n");
        printf("...allocating CPU memory\n");
        cutilSafeMalloc( h_Kernel    = (float *)malloc(KERNEL_SIZE) );
        cutilSafeMalloc( h_Data      = (float *)malloc(DATA_SIZE)   );
        cutilSafeMalloc( h_ResultCPU = (float *)malloc(DATA_SIZE)   );
        cutilSafeMalloc( h_ResultGPU = (float *)malloc(DATA_SIZE)   );
        printf("...allocating GPU memory\n");
        cutilSafeCall( cudaMalloc((void **)&d_Kernel, DATA_SIZE) );
        cutilSafeCall( cudaMalloc((void **)&d_Data,   DATA_SIZE) );

        printf("...generating data\n");
        printf("Data length: %i; kernel length: %i\n", dataN, kernelN);
        srand(2007);
        for (i = 0; i < kernelN; i++)
            h_Kernel[i] = (float)rand() / (float)RAND_MAX;

        for (i = 0; i < dataN; i++)
            h_Data[i] = (float)rand() / (float)RAND_MAX;

        cutilSafeCall( cudaMemset(d_Kernel, 0, DATA_SIZE) );
        cutilSafeCall( cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy(d_Data,   h_Data,     DATA_SIZE, cudaMemcpyHostToDevice) );

    printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");
    cutilSafeCall( cutilDeviceSynchronize() );
    cutilCheckError( cutResetTimer(hTimer) );
    cutilCheckError( cutStartTimer(hTimer) );
        fwtBatchGPU(d_Data, 1, log2Data);
        fwtBatchGPU(d_Kernel, 1, log2Data);
        modulateGPU(d_Data, d_Kernel, dataN);
        fwtBatchGPU(d_Data, 1, log2Data);
    cutilSafeCall( cutilDeviceSynchronize() );
    cutilCheckError( cutStopTimer(hTimer) );
    gpuTime = cutGetTimerValue(hTimer);
    printf("GPU time: %f ms; GOP/s: %f\n", gpuTime, NOPS / (gpuTime * 0.001 * 1E+9));

    printf("Reading back GPU results...\n");
    cutilSafeCall( cudaMemcpy(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost) );

    printf("Running straightforward CPU dyadic convolution...\n");
    dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

    printf("Comparing the results...\n");
        sum_delta2 = 0;
        sum_ref2   = 0;
        for(i = 0; i < dataN; i++){
            delta       = h_ResultCPU[i] - h_ResultGPU[i];
            ref         = h_ResultCPU[i];
            sum_delta2 += delta * delta;
            sum_ref2   += ref * ref;
        }
        L2norm = sqrt(sum_delta2 / sum_ref2);
        printf("L2 norm: %E\n", L2norm);
    printf((L2norm < 1e-6) ? "PASSED\n" : "FAILED\n");


    printf("Shutting down...\n");
        cutilCheckError(  cutDeleteTimer(hTimer) );
        cutilSafeCall( cudaFree(d_Data)   );
        cutilSafeCall( cudaFree(d_Kernel) );
        free(h_ResultGPU);
        free(h_ResultCPU);
        free(h_Data);
        free(h_Kernel);

    cutilDeviceReset();
    printf("L2 norm: %E\n", L2norm);
    shrQAFinishExit(argc, (const char **)argv, (L2norm < 1e-6) ? QA_PASSED : QA_FAILED);
}