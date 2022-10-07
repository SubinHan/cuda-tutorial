#include <stdio.h>

#define N 65536

__global__ void device_add(int *a, int *b, int *c) 
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void fill_array(int *data)
{
	for(int idx = 0; idx < N; idx++)
		data[idx] = idx;
}

void print_output(int *a, int *b, int *c)
{
	for(int idx = 0; idx < N; idx++)
		printf("\n %d + %d = %d", a[idx], b[idx], c[idx]);
}

__global__ void device_print_output(int *a, int *b, int *c)
{
	printf("\n %d + %d = %d", a[blockIdx.x], b[blockIdx.x], c[blockIdx.x]);
}

int main(void) {
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;
	int size = N * sizeof(int);
	
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); fill_array(a);
	b = (int *)malloc(size); fill_array(b);
	c = (int *)malloc(size);
	
	cudaMalloc((void **)&dev_a, size);
	cudaMalloc((void **)&dev_b, size);
	cudaMalloc((void **)&dev_c, size);
	
	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
	device_add<<<N,1>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
	
	//print_output(a, b, c);
	device_print_output<<<N, 1>>>(dev_a, dev_b, dev_c);
	
	free(a); free(b); free(c);
	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
	
	return 0;
}
