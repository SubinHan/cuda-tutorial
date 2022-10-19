#include <stdio.h>

#define LENGTH 14

__global__ void MatrixAdd( int*M, int*N, int*P )
{
	int tid, tx, ty;
	tx = blockDim.x * blockIdx.x + threadIdx.x;
	ty = blockDim.y * blockIdx.y + threadIdx.y;
	tid = gridDim.x * blockDim.x * ty + tx;
	P[tid] = M[tid] + N[tid];
}

void printResult(int* M, int* N, int* P)
{
	int row = 0; int col = 0;
	for (row = 0; row < LENGTH; row++)
	{
		for (col = 0; col < LENGTH; col++)
		{
			int Destindex = row * LENGTH + col;
			printf( "%d (= P[%d][%d]) = %d (= M[%d][%d]) + %d (= N[%d][%d]) \n", 
			P[Destindex], row, col, M[Destindex], row, col, N[Destindex], row, col );
		}
	}
}

int main()
{
	const int MatrixSize = LENGTH * LENGTH;
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
	
	dim3 Dg(3, 3, 1);
	dim3 Db(8, 6, 1);
	
	MatrixAdd <<<Dg,Db>>> (dev_M, dev_N, dev_P);
	
	cudaMemcpy(P, dev_P, BufferSize, cudaMemcpyDeviceToHost);
	
	printResult(M, N, P);
	
	cudaFree(dev_M); cudaFree(dev_N); cudaFree(dev_P);
	free(M); free(N); free(P);
	
	return 0;
}
