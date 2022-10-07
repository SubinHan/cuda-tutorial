#include <stdio.h>

__global__ void device_add(int *a, int *b, int *c)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

void fill_array(int *data, int length)
{
	for(int idx = 0; idx < length; idx++)
		data[idx] = idx;
}

int main()
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int arr_cnt;
	
	printf("type the array conut: ");
	scanf("%d", &arr_cnt);
	
	int size = arr_cnt * sizeof(int);
	
	a = (int *)malloc(size); fill_array(a, arr_cnt);
	b = (int *)malloc(size); fill_array(b, arr_cnt);
	c = (int *)malloc(size);
	
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	int threads_per_block = 1024;
	int div = arr_cnt/threads_per_block;
	printf("arr_cnt/threads_per_block = %d \n", div);
	
	device_add<<<div+1, threads_per_block>>>(d_a,d_b,d_c);
	
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	
	bool success= true;
	for(int i = 0; i < arr_cnt; i++)
	{
		if(a[i] + b[i] != c[i])
		{
			printf("Erorr: %d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}
	if(success)
		printf("We did it!\n");
	
	free(a); free(b); free(c); 
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	
	return 0;
}

