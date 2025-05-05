/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512
#define SIMPLE


    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

__global__ void reduction(float *out, float *in, unsigned size)
{


    // INSERT KERNEL CODE HERE

    // Using shared memory instead of Global memory like in Book
    __shared__ float sdata[BLOCK_SIZE];

    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BLOCK_SIZE * 2) + tid;

    float sum = 0.0f;

    if (i < size) {
        sum = in[i];
        if (i + BLOCK_SIZE < size)
            sum += in[i + BLOCK_SIZE];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
