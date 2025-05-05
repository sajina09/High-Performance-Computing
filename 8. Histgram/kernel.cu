/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE



#include <cuda.h>

// Non optmiized version
__global__ void histogram_naive_kernel(unsigned int *input, uint8_t *bins, unsigned int num_elements, unsigned int num_bins) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        unsigned int bin = input[idx];
        if (bin < num_bins) {
            // Atomic add directly to uint8_t bin
            // Cast to unsigned int pointer to perform atomicAdd
            atomicAdd((unsigned int*)&bins[bin], 1);
        }
    }
}

// void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements, unsigned int num_bins) {
//     // Launch simple kernel directly
//     const unsigned int blockSize = 256;
//     unsigned int gridSize = (num_elements + blockSize - 1) / blockSize;

//     histogram_naive_kernel<<<gridSize, blockSize>>>(input, bins, num_elements, num_bins);

//     cudaDeviceSynchronize();
// }


//------------------------------- Ends here -----------------------------------

// Count things carefully into a safe (big) format
// Building the counts of how many times each number appears.
__global__ void histogram_kernel(unsigned int *input, unsigned int *bins_temp, unsigned int num_elements, unsigned int num_bins) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        unsigned int bin = input[idx];
        if (bin < num_bins) {
            // Safe atomic add on unsigned int
            atomicAdd(&bins_temp[bin], 1);
        }
    }
}

// Clamp the counts to 255 and shrink them into 8-bit bins
// After the counting is finished, each thread looks at one bin from bins_temp[].
// checks if the count is more than 255. if > 255 it saturates (sets it to 255).
__global__ void saturate_kernel(unsigned int *bins_temp, uint8_t *bins, unsigned int num_bins) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_bins) {
        unsigned int val = bins_temp[idx];
        if (val > 255) val = 255;
        bins[idx] = (uint8_t)val;
    }
}

// Optmization 2 
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins_temp, unsigned int num_elements, unsigned int num_bins) {
    extern __shared__ unsigned int local_bins[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory histogram
    for (int i = tid; i < num_bins; i += blockDim.x) {
        local_bins[i] = 0;
    }
    __syncthreads();

    // Update local shared memory bins
    if (idx < num_elements) {
        unsigned int bin = input[idx];
        if (bin < num_bins) {
            atomicAdd(&local_bins[bin], 1);
        }
    }
    __syncthreads();

    // Write shared memory bins to global memory bins
    for (int i = tid; i < num_bins; i += blockDim.x) {
        if (local_bins[i] > 0) {
            atomicAdd(&bins_temp[i], local_bins[i]);
        }
    }
}


/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
// void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements,
//         unsigned int num_bins) {

//     // INSERT CODE HERE

//      // Allocate temporary bins (unsigned int) on the device
//      unsigned int* bins_temp;
//      cudaMalloc((void**)&bins_temp, num_bins * sizeof(unsigned int));
 
//      // Initialize bins_temp to 0
//      cudaMemset(bins_temp, 0, num_bins * sizeof(unsigned int));
 
//      // Launch histogram kernel
//      const unsigned int blockSize = 256;
//      unsigned int gridSize_input = (num_elements + blockSize - 1) / blockSize;
 
//      histogram_kernel<<<gridSize_input, blockSize>>>(input, bins_temp, num_elements, num_bins);
 
//      // Wait for histogram computation to finish
//      cudaDeviceSynchronize();
 
//      // Launch saturate kernel to convert to uint8_t
//      unsigned int gridSize_bins = (num_bins + blockSize - 1) / blockSize;
 
//      saturate_kernel<<<gridSize_bins, blockSize>>>(bins_temp, bins, num_bins);
 
//      cudaDeviceSynchronize();
 
//      // Free temporary memory
//      cudaFree(bins_temp);

// }

// 2----------------

void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements, unsigned int num_bins) {
    // Allocate temporary bins (unsigned int) on device
    unsigned int* bins_temp;
    cudaMalloc((void**)&bins_temp, num_bins * sizeof(unsigned int));

    // Initialize bins_temp to 0
    cudaMemset(bins_temp, 0, num_bins * sizeof(unsigned int));

    // Launch histogram kernel with shared memory
    const unsigned int blockSize = 256;
    unsigned int gridSize_input = (num_elements + blockSize - 1) / blockSize;

    // Calculate shared memory size needed
    size_t sharedMemSize = num_bins * sizeof(unsigned int);

    histogram_shared_kernel<<<gridSize_input, blockSize, sharedMemSize>>>(input, bins_temp, num_elements, num_bins);

    cudaDeviceSynchronize();

    // Launch saturate kernel to convert to uint8_t
    unsigned int gridSize_bins = (num_bins + blockSize - 1) / blockSize;

    saturate_kernel<<<gridSize_bins, blockSize>>>(bins_temp, bins, num_bins);

    cudaDeviceSynchronize();

    // Free temporary memory
    cudaFree(bins_temp);
}

 



