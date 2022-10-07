#include <stdio.h>

#define N 32

__global__ void device_add(int *a, int *b, int *c, int *id)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	id[index] = index;
	c[index] = a[index] + b[index];
	printf("[thread id: %d] = [threadIdx.x: %d] & [blockIdx.x: %d]\n", index, threadIdx.x, blockIdx.x);
}

void fill_array(int *data)
{
	for(int idx = 0; idx < N; idx++)
		data[idx] = idx;
}

void print_output(int *a, int *b, int *c, int *id)
{
	for(int idx = 0; idx < N; idx++)
		printf("\n [thread id: %d] %d + %d = %d", id[idx], a[idx], b[idx], c[idx]);
}

int main()
{
	int *a, *b, *c, *id;
	int *d_a, *d_b, *d_c, *d_id;
	int threads_per_block=0, no_of_blocks=0;
	
	int size = N * sizeof(int);
	
	a = (int *)malloc(size); fill_array(a);
	b = (int *)malloc(size); fill_array(b);
	c = (int *)malloc(size);
	id = (int *)malloc(size);
	
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	cudaMalloc((void **)&d_id, size);
	
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	threads_per_block = 4;
	no_of_blocks = N/threads_per_block; 
	device_add<<<no_of_blocks,threads_per_block>>>(d_a,d_b,d_c, d_id);
	
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(id, d_id, size, cudaMemcpyDeviceToHost);
	
	print_output(a,b,c,id);
	
	free(a); free(b); free(c); free(id);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_id);
	
	return 0;
}

