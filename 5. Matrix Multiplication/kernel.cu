/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for SGEMM (Single-Precision General Matrix Multiplication)
__global__ void mysgemm(int m, int n, int k, const float *A,
                        const float *B, float *C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {  // Ensure within matrix bounds
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];  // Row-major order
        }
        C[row * n + col] = sum;  // Store result in C
    }
} // End of mysgemm kernel

// Wrapper function to call CUDA kernel
extern "C" void basicSgemm(char transa, char transb, int m, int n, int k,
                           float alpha, const float *A, int lda, const float *B,
                           int ldb, float beta, float *C, int ldc) {
    
    if ((transa != 'N') && (transa != 'n')) {
        printf("Error: unsupported value of 'transa'\n");
        return;
    }

    if ((transb != 'N') && (transb != 'n')) {
        printf("Error: unsupported value of 'transb'\n");
        return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
        printf("Error: unsupported value of alpha\n");
        return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
        printf("Error: unsupported value of beta\n");
        return;
    }

    // Define CUDA thread block size
    const unsigned int BLOCK_SIZE = 16; // 16x16 threads per block

    // Compute grid dimensions (rounding up properly)
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Launch the CUDA kernel
    mysgemm<<<gridDim, blockDim>>>(m, n, k, A, B, C);

    // Ensure kernel execution completes before returning
    cudaDeviceSynchronize();
} // End of basicSgemm function
