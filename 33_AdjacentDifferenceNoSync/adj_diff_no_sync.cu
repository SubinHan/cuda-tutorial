#include <stdio.h>
#include "../Common/Timer.h"

#define SIZE 2048
#define BLOCK_SIZE 128

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

__global__ void adj_diff_shared(int* result, int* input)
{
	int tx = threadIdx.x;
	__shared__ int s_data[BLOCK_SIZE];
	
	// each thread reads one element to s_data
	unsigned int i = blockDim.x * blockIdx.x + tx;
	s_data[tx] = input[i];
	
	// avoid race condition: ensure all loads
	// complete before continuing
	//__syncthreads();
	
	if(tx > 0) 
		result[i] = s_data[tx] + s_data[tx-1];
	else if(i > 0) 
		result[i] = s_data[tx] + input[i-1];
}

int main()
{
	const int size = SIZE;
	const int BufferSize = size*sizeof(int);
	
	struct timeval naive_start, naive_end;
	struct timeval shared_start, shared_end;
	
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
	
	int num_block = SIZE / BLOCK_SIZE;
	
	gettimeofday(&naive_start, NULL);
	adj_diff_naive<<<num_block, BLOCK_SIZE>>>(dev_Out, dev_In);
	gettimeofday(&naive_end, NULL);
	
	gettimeofday(&shared_start, NULL);
	adj_diff_shared<<<num_block, BLOCK_SIZE>>>(dev_Out, dev_In);
	gettimeofday(&shared_end, NULL);
	
	struct timeval naive_gap, shared_gap;
	getGapTime(&naive_start, &naive_end, &naive_gap); 
	getGapTime(&shared_start, &shared_end, &shared_gap); 
	float f_naive_gap = timevalToFloat(&naive_gap);
	float f_shared_gap = timevalToFloat(&shared_gap);
	
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
	
	printf("naive: %f\n shared: %f\n", f_naive_gap, f_shared_gap);
	
	cudaFree(dev_In); cudaFree(dev_Out);
	free(Input); free(Output);
	
	return 0;
}
