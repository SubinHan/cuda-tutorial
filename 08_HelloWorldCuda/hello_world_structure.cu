#include <stdio.h>

__global__ void print_from_gpu(void)
{
  printf("Hello, World! from thread [%d, %d] on device! \n", threadIdx.x, blockIdx.x);
}

int main()
{
  printf("Hello, World! from host! \n");
  print_from_gpu<<<5,3>>>();
  cudaDeviceSynchronize();
  
  getchar();
  
  return 1;
}