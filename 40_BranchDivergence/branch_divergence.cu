#include <stdio.h>
#include "../Common/Timer.h"

__global__ void rearrange(int*input, int*result, int num_of_elements)
{
	int tid, gx;
	tid = threadIdx.x;
	gx = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid % 2 == 0)
	{
		result[gx/2] = input[gx];
	}
	else
	{
		result[num_of_elements/2 + gx/2] = input[gx];
	}
}

__global__ void rearrange_warp(int*input, int*result, int num_of_elements)
{
	int gx;
	gx = blockDim.x * blockIdx.x + threadIdx.x;
	if(gx < num_of_elements/2)
	{
		result[gx] = input[gx*2];
	}
	else
	{
		result[gx] = input[(gx - num_of_elements/2) * 2 + 1];
	}
}

void print1DArray(int* result, int num_of_elements)
{
	for (int i = 0; i < num_of_elements; i++){
		printf( "%d ", result[i]);
	}
	printf( "\n");
}


int main()
{
	srand(time(NULL));
	struct timeval htod_start, htod_end;
	struct timeval gpu_start, gpu_end;
	struct timeval dtoh_start, dtoh_end;
	struct timeval init_start, init_end, init_gap;
	struct timeval start, end, gap;
	
	int* input; int* result;
	int BufferSize;
	int* dev_input; int* dev_result;
	int grid_dim;
	int block_dim=512;

	for(int num_of_elements=block_dim; num_of_elements<1024*1024; num_of_elements*=2)
	{
		gettimeofday(&start, NULL);
		
		grid_dim = num_of_elements / block_dim;
		BufferSize = num_of_elements * sizeof(int);
		
		input = (int*)malloc(BufferSize);
		result = (int*)malloc(BufferSize);
		
		gettimeofday(&init_start, NULL);
		for(int i=0; i<num_of_elements; i++)
		{
			input[i] = i;
		}
		gettimeofday(&init_end, NULL);
		
		cudaMalloc((void**)&dev_input, BufferSize);
		cudaMalloc((void**)&dev_result, BufferSize);
		
		gettimeofday(&htod_start, NULL);
		cudaMemcpy(dev_input, input, BufferSize, cudaMemcpyHostToDevice);
		gettimeofday(&htod_end, NULL);
		
		struct timeval htod_gap;
		getGapTime(&htod_start, &htod_end, &htod_gap);

		dim3 Dg(grid_dim, 1, 1);
		dim3 Db(block_dim, 1, 1);
		
		gettimeofday(&gpu_start, NULL);
		rearrange <<<Dg,Db>>> (dev_input, dev_result, num_of_elements);
		// rearrange_warp <<<Dg,Db>>> (dev_input, dev_result, num_of_elements);
		cudaDeviceSynchronize();
		gettimeofday(&gpu_end, NULL);
		
		gettimeofday(&dtoh_start, NULL);
		cudaMemcpy(result, dev_result, BufferSize, cudaMemcpyDeviceToHost);
		gettimeofday(&dtoh_end, NULL);
		
		struct timeval gpu_gap, dtoh_gap;
		getGapTime(&gpu_start, &gpu_end, &gpu_gap);
		getGapTime(&dtoh_start, &dtoh_end, &dtoh_gap);
		
		float f_htod_gap = timevalToFloat(&htod_gap);
		float f_gpu_gap = timevalToFloat(&gpu_gap);
		float f_dtoh_gap = timevalToFloat(&dtoh_gap);
		
		float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;
		getGapTime(&init_start, &init_end, &init_gap);
		
		float f_init_gap = timevalToFloat(&init_gap);
		
		gettimeofday(&end, NULL);
		getGapTime(&start, &end, &gap);
		float f_gap = timevalToFloat(&gap);
		
		printf("[Cuda] num_of_elements = %d, total time = %.6f, htod time = %.6f, GPU time = %.6f, dtoh time = %.6f \n",
		num_of_elements, total_gap, f_htod_gap, f_gpu_gap, f_dtoh_gap);
		
		printf("[Result] even_first = %d, even_last = %d, odd_first = %d, odd_last = %d \n",
		result[0], result[num_of_elements/2-1], result[num_of_elements/2], result[num_of_elements-1]);
		
		cudaFree(dev_input); cudaFree(dev_result);
		free(input); free(result);
	}
	return 0;
}
