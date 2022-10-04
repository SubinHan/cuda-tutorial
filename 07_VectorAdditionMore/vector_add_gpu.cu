#include <stdio.h>

#define N 20
#define DIM 4

__global__ void add(int *a, int *b, int *c, int *d, int *e)
{
  int tid = blockIdx.x;
  e[tid] = a[tid] + b[tid] + c[tid] + d[tid];
}

int main()
{
  int size = N * sizeof(int);

  int a[N], b[N], c[N], d[N], e[N];

  int *dev_a, *dev_b, *dev_c, *dev_d, *dev_e;

  cudaMalloc((void**)&dev_a, size);
  cudaMalloc((void**)&dev_b, size);
  cudaMalloc((void**)&dev_c, size);
  cudaMalloc((void**)&dev_d, size);
  cudaMalloc((void**)&dev_e, size);

  for(int i = 0; i < N; i++)
  {
    a[i] = i;
    b[i] = 10 * i;
    c[i] = 100 * i;
    d[i] = 1000 * i;
  }

  cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_d, d, size, cudaMemcpyHostToDevice);

  add<<<N,1>>>(dev_a, dev_b, dev_c, dev_d, dev_e);

  cudaMemcpy(e, dev_e, size, cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cudaFree(dev_d);
  cudaFree(dev_e);

  for(int i = 0; i < N; i++)
  {
    printf("%d + %d + %d + %d = %d\n", a[i], b[i], c[i], d[i], e[i]);
  }
  
  getchar();

  return 1;
}