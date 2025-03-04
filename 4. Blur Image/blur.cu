/******************************************************************************
 *cr
 *cr            (C) Copyright 2025 The Board of Trustees of the
 *cr                        Florida Institute of Technology
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define STB implementations before including headers
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel for 3x3, 5x5, and 9x9 blur (each pixel is replaced by the average of its neighbors)
__global__ void blurKernel(const unsigned char *input, unsigned char *output, int width, int height)
{
    //INSERT CODE HERE

    // Calculate global thread index based on the block and thread indices x and y ----

    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Instead of x and y I used col and row for clarification 

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check the boundary conditions of both indices x and y ---

    // Use global index to determine which elements to read, add, and write ---

    // INSERT KERNEL CODE HERE

    // Update the output

    if (row < height && col < width) {
        // BLUR_RADIUS = 1 for 3 x 3, 2 for 5 x 5, and 4 for 9 x 9.
        const int BLUR_RADIUS = 4;
        unsigned int idx = row * width + col;
        int sum = 0;
        int count = 0;

        for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; i++) {
            for (int j = -BLUR_RADIUS; j <= BLUR_RADIUS; j++) {
                int inRow = row + i;
                int inCol = col + j;

                if (inRow < height && inRow >= 0 && inCol < width && inCol >= 0) {
                    sum += input[inRow * width + inCol];
                    count++;
                    // printf("Sum: %d, Count: %d\n", sum, count++);

                }
            }
        }

        output[idx] = sum / count;

        // if (idx < 10) printf("output[%d] = %d\n", idx, output[idx]);
        // printf("output[%d] = %d\n", idx, output[idx]);

    }

}

int main()
{
    int width, height, channels;

    // Load image from disk; force 1 channel (grayscale)
    unsigned char *h_input = stbi_load("input.png", &width, &height, &channels, 1);
    if (h_input == NULL) {
        fprintf(stderr, "Error: could not load image\\n");
        return 1;
    }


    cudaError_t cuda_ret;
    // Initialize host variables ----------------------------------------------

    //INSERT CODE HERE
    unsigned char *h_output;
    h_output = (unsigned char*) malloc(width * height * sizeof(unsigned char));

    // putting the device variable here.
    unsigned char *d_input, *d_output;

    // Allocate device variables ----------------------------------------------

    //INSERT CODE HERE
     // Row majour linearized input image 
    size_t imgSize = width * height * sizeof(unsigned char);

    cudaMalloc((void**)&d_input, imgSize);
    cudaMalloc((void**)&d_output, imgSize);

    // Copy input image data to device memory
    //INSERT CODE HERE
    cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_output, h_output, imgSize, cudaMemcpyHostToDevice);

     // Define block and grid sizes
    dim3 block(16, 16, 1);
    dim3 grid(ceil(width/16.0), ceil(height/16.0), 1);

    // Launch the blur kernel
    printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
    printf("Block: %d, %d, %d\n", block.x, block.y, block.z);
    
    // Launch kernel ----------------------------------------------------------
    //INSERT CODE HERE

    blurKernel<<<grid, block>>>(d_input, d_output, width, height);

    
    if (cuda_ret != cudaSuccess)
    {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }

    // Wait for the GPU to finish -----------------------------------------------
    cudaDeviceSynchronize();

    // Copy the blurred image back to host memory -------------------------------
    //INSERT CODE HERE

    cudaMemcpy(h_input, d_input, imgSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();


    // Save the output image ----------------------------------------------------
    // stbi_write_png("output.png", width, height, 1, h_output, width);
    // stbi_write_png("output.png", width, height, 1, h_output, width * sizeof(unsigned char));
    cudaDeviceSynchronize();
    printf("Start exporting -->> \n");

    cuda_ret = cudaGetLastError();  // Get the last error
    if (cuda_ret != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(cuda_ret));
    } else {
    printf("No CUDA errors detected.\n");
    }

    if (!stbi_write_png("output.png", width, height, 1, h_output, width)) {
    printf("Failed to write image\n");
    } else {
    printf("Image written successfully\n");
    }


    printf("Output image saved");

    // Free device and host memory ----------------------------------------------
    //INSERT CODE HERE
    cudaFree(d_input);
    cudaFree(d_output);


    stbi_image_free(h_input);
    free(h_output);

    printf("Image blur completed. Output saved as output.png\n");
    return 0;
}
