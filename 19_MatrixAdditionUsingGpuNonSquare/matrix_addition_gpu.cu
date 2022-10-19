#include <stdio.h>

#define MIN_NUM_THREAD 16

__global__ void MatrixAdd( int*M, int*N, int*P )
{
	int tid, tx, ty;
	tx = blockDim.x * blockIdx.x + threadIdx.x;
	ty = blockDim.y * blockIdx.y + threadIdx.y;
	tid = gridDim.x * blockDim.x * ty + tx;
	P[tid] = M[tid] + N[tid];
}

void printResult(int* M, int* N, int* P, int width, int height)
{
	int row = 0; int col = 0;
	for (row = 0; row < height; row++)
	{
		for (col = 0; col < width; col++)
		{
			int Destindex = row * width + col;
			printf( "%d (= P[%d][%d]) = %d (= M[%d][%d]) + %d (= N[%d][%d]) \n", 
			P[Destindex], row, col, M[Destindex], row, col, N[Destindex], row, col );
		}
	}
}

int main()
{
	int width; int height;
	printf("type the width: ");
	scanf("%d", &width);
	printf("type the height: ");
	scanf("%d", &height);

	const int MatrixSize = width * height;
	const int BufferSize = MatrixSize * sizeof(int);
	
	int* M; int* N; int* P;
	
	M = (int*)malloc(BufferSize);
	N = (int*)malloc(BufferSize);
	P = (int*)malloc(BufferSize);
	
	for( int i = 0; i < MatrixSize; i++)
	{
		M[i] = i; N[i] = i; P[i] = 0;
	}
	
	int* dev_M; int* dev_N; int* dev_P;
	
	cudaMalloc((void**)&dev_M, BufferSize);
	cudaMalloc((void**)&dev_N, BufferSize);
	cudaMalloc((void**)&dev_P, BufferSize);
	
	cudaMemcpy(dev_M, M, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_N, N, BufferSize, cudaMemcpyHostToDevice);
	
	int blockDimX = MIN_NUM_THREAD;
	int blockDimY = MIN_NUM_THREAD;
	int gridDimX = width / MIN_NUM_THREAD + 1;
	int gridDimY = height / MIN_NUM_THREAD + 1;
	
	dim3 Dg(gridDimX, gridDimY, 1);
	dim3 Db(blockDimX, blockDimY, 1);
	
	MatrixAdd <<<Dg,Db>>> (dev_M, dev_N, dev_P);
	
	cudaMemcpy(P, dev_P, BufferSize, cudaMemcpyDeviceToHost);
	
	printResult(M, N, P, width, height);
	
	cudaFree(dev_M); cudaFree(dev_N); cudaFree(dev_P);
	free(M); free(N); free(P);
	
	return 0;
}
