/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include "support.cu"
#include "kernel.cu"



int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------
    float *dev_a, *dev_b, *dev_c;  // Device pointers
 
    printf("\nSetting up the problem..."); 
    fflush(stdout);
    startTime(&timer);

    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }

    float* A_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { A_h[i] = (rand()%100)/100.00; }

    float* B_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { B_h[i] = (rand()%100)/100.00; }

    float* C_h = (float*) malloc( sizeof(float)*n );


    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); 
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMalloc((void**) &dev_a, n * sizeof(float));
    cudaMalloc((void**) &dev_b, n * sizeof(float));
    cudaMalloc((void**) &dev_c, n * sizeof(float));



    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); 
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE


    cudaMemcpy(dev_a, A_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, B_h, n * sizeof(float), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); 
    fflush(stdout);
    startTime(&timer);

    

    //INSERT CODE HERE
    vecAddKernel<<<ceil(n / 256.0), 256>>>(dev_a, dev_b, dev_c, n);
    cuda_ret = cudaDeviceSynchronize();


    if(cuda_ret != cudaSuccess)
      FATAL("Unable to launch kernel");
    
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // Print this to check the PTX error.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); 
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(C_h, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Printing the value of C to check the values
    // printf("C_h --------values:\n");
    // for (unsigned int i = 0; i < 10 && i < n; i++) {
    //     printf("C_h[%d] = %f\n", i, C_h[i]);
    // }


    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); 
    fflush(stdout);

    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

