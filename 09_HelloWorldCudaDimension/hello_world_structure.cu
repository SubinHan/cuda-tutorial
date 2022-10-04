#include <stdio.h>

__global__ void print_from_gpu(void)
{
  printf("Hello, World! from thread [%d, %d, %d, %d] on device! \n", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.z);
}

int main()
{
  printf("Hello, World! from host! \n");
  dim3 Dg(3, 2, 3);
  print_from_gpu<<<Dg,1>>>();
  cudaDeviceSynchronize();
  
  return 1;
}