/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#ifndef KERNEL_CU__
#define KERNEL_CU__

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {

    // Calculate global thread index based on the block and thread indices ----
    //INSERT KERNEL CODE HERE
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Use global index to determine which elements to read, add, and write ---
    //INSERT KERNEL CODE HERE
    if(idx < n){
        C[idx] = A[idx] + B[idx];
        // printf("GPU: C[%d] = %f\n", idx, C[idx]);  // ✅ Print inside GPU
    }
}

#endif