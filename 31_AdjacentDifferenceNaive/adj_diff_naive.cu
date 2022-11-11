#include <stdio.h>

#define SIZE 2048

// motivate shared variables with
// Adjacent Difference application:
// compute result[i] = input[i] – input[i-1]
__global__ void adj_diff_naive(int *result, int *input)
{
	// compute this thread’s global index
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i > 0)
	{
		// each thread loads two elements from global memory
		int x_i = input[i];
		int x_i_minus_one = input[i-1];
		
		result[i] = x_i + x_i_minus_one;
	}
}

int main()
{
	const int size = SIZE;
	const int BufferSize = size*sizeof(int);
	
	int* Input; int* Output;
	
	Input = (int*)malloc(BufferSize);
	Output = (int*)malloc(BufferSize);
	
	int i = 0;
	
	for(i = 0; i < size; i++)
	{
		Input[i] = i; Output[i] = 0;
	}
	
	int* dev_In; int* dev_Out;
	
	cudaMalloc((void**)&dev_In, size*sizeof(int));
	cudaMalloc((void**)&dev_Out, size*sizeof(int));
	
	cudaMemcpy(dev_In, Input, size*sizeof(int), cudaMemcpyHostToDevice);
	
	adj_diff_naive<<<32,64>>>(dev_Out, dev_In);
	
	cudaMemcpy(Output, dev_Out, size*sizeof(int), cudaMemcpyDeviceToHost);
	
	for(i = 0; i < 5; i++)
	{
		printf(" Output[%d] : %d\n",i,Output[i]);
	}
	printf(" ......\n");
	for(i = size-5; i < size; i++)
	{
		printf(" Output[%d] : %d\n",i,Output[i]);
	}
	
	for(int i = 1; i < size; i++)
	{
		if(Input[i-1] + Input[i] != Output[i])
		{
			printf("failed at %d", i);
			break;
		}
	}
	
	cudaFree(dev_In); cudaFree(dev_Out);
	free(Input); free(Output);
	
	return 0;
}
