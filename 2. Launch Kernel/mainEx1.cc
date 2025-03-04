#include <cstdio>

__global__ void myKernel() {
  // The kernel will not execute any code.
}

int main() {
  myKernel<<<1, 1>>>(); // Launching the kernel from main()

  printf("Hello World!\n");
  return 0;
}