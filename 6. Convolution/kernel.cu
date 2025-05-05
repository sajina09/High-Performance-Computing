/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
    // Debug message (optional)
    // printf("--- Kernel ---\n");

    // Shared memory tile (block-sized + halo)
    __shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE];
    
    // Compute global indices (output pixel coordinates)
    int col_o = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row_o = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Compute input indices (including halo)
    int col_i = col_o - FILTER_SIZE / 2;
    int row_i = row_o - FILTER_SIZE / 2;

    // Load input elements into shared memory with boundary checking
    if (row_i >= 0 && row_i < N.height && col_i >= 0 && col_i < N.width) {
        N_s[threadIdx.y][threadIdx.x] = N.elements[row_i * N.width + col_i];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f; // zero-padding outside image boundary
    }

    // Synchronize all threads within block before computing convolution
    __syncthreads();

    // Only compute for threads inside the TILE region (avoid computing in the halo area)
    if (threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE) {

        // Only output if within bounds of output image
        if (row_o < P.height && col_o < P.width) {
            float sum = 0.0f;

            // Apply convolution filter
            for (int i = 0; i < FILTER_SIZE; ++i) {
                for (int j = 0; j < FILTER_SIZE; ++j) {
                    sum += M_c[i][j] * N_s[threadIdx.y + i][threadIdx.x + j];
                }
            }

            // Write the computed value to output matrix
            P.elements[row_o * P.width + col_o] = sum;
        }
    }
}
