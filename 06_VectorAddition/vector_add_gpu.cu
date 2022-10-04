#include <stdio.h>

#define N 20

__global__ void add(int *a, int *b, int *c)
{
  int tid = blockIdx.x;
  c[tid] = a[tid] + b[tid];
}

int main()
{
  int size = N * sizeof(int);

  int a[N], b[N], c[N];

  int *dev_a, *dev_b, *dev_c;

  cudaMalloc((void**)&dev_a, size);
  cudaMalloc((void**)&dev_b, size);
  cudaMalloc((void**)&dev_c, size);

  for(int i = 0; i < N; i++)
  {
    a[i] = i;
    b[i] = 100 * i;
  }

  cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

  add<<<N,1>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  for(int i = 0; i < N; i++)
  {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  return 1;
}