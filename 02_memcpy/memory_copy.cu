#include <stdio.h>

int main()
{
  const int size = 10;
  int src_data[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int dst_data[size] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  printf("src_data = ");
  for(int i = 0; i < size; i++)
    printf("%d ", src_data[i]);
  printf("\n");

  int* dev_src = 0;
  int* dev_dst = 0;

  cudaMalloc((void**)&dev_src, size * sizeof(int));
  cudaMalloc((void**)&dev_dst, size * sizeof(int));

  cudaMemcpy(dev_src, src_data, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dst, dev_src, size * sizeof(int), cudaMemcpyDeviceToDevice);
  cudaMemcpy(dst_data, dev_dst, size * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dev_src);
  cudaFree(dev_dst);

  printf("dst_data = ");
  for(int i = 0; i < size; i++)
    printf("%d ", dst_data[i]);
  printf("\n");

  return 1;
}