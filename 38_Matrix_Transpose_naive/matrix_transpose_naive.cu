#include <stdio.h>
#include "../Common/Timer.h"

#define MATRIX_SIZE 1000

__global__ void matrix_transpose_naive(int *input, int *output, int WIDTH) {
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	
	int index = indexY * WIDTH + indexX;
	int transposedIndex = indexX * WIDTH + indexY; 
	
	output[transposedIndex] = input[index];
	
	printf(
		"[cuda] threadIdx.x,y: (%d,%d), index: %d, input[index]_addr: %p, transposedIndex: %d, output[transposedIndex]_addr: %p\n", 
		threadIdx.x, threadIdx.y, index, &input[index], transposedIndex, &output[transposedIndex]);
}

void printResult(int* M, int WIDTH)
{
	printf( "\n");
	for (int row = 0; row < WIDTH; row++)
	{
		for (int col = 0; col < WIDTH; col++)
		{
			int index = row * WIDTH + col;
			printf("%d ", M[index]);
		}
		printf( "\n");
	}
	printf( "\n");
}

void make_matrix(int* matrix)
{
	int size = MATRIX_SIZE * MATRIX_SIZE;
	for(int i = 0; i < size; i++)
	{
		matrix[i] = rand() % 8;
	}
}

int main()
{
	int WIDTH = 5;
	
	int bufferSize = WIDTH*WIDTH;
	
	int* input; int* output;
	int* dev_input; int* dev_output;
	
	input = (int*)malloc(bufferSize*sizeof(int));
	output = (int*)malloc(bufferSize*sizeof(int));
	
	for(int i=0; i<bufferSize; i++){
		input[i] = i;
		output[i] = 0;
	}
	
	printResult(input, WIDTH);
	
	cudaMalloc((void**)&dev_input, bufferSize*sizeof(int));
	cudaMalloc((void**)&dev_output, bufferSize*sizeof(int));
	
	cudaMemcpy(dev_input, input, bufferSize*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_output, output, bufferSize*sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 Dg(1, 1, 1); dim3 Db(5, 5, 1);
	matrix_transpose_naive<<<Dg,Db>>> (dev_input, dev_output, WIDTH);
	cudaDeviceSynchronize();
	
	cudaMemcpy(output, dev_output, bufferSize*sizeof(int), cudaMemcpyDeviceToHost);
	
	printResult(output, WIDTH);
	
	cudaFree(dev_input); cudaFree(dev_output);
	free(input); free(output);
	
	return 0;
}
