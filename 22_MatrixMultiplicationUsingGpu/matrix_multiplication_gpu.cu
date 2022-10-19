#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LENGTH 4

__global__ void MatrixMulCuda( int*M, int*N, int*P )
{
	int tid, tx, ty;
	tx = blockDim.x * blockIdx.x + threadIdx.x;
	ty = blockDim.y * blockIdx.y + threadIdx.y;
	
	int DimX = gridDim.x * blockDim.x;
	tid = DimX * ty + tx;
	
	int Value = 0; int MVal = 0; int NVal = 0;
	for (int k = 0; k < LENGTH; k++)
	{
		MVal = M[ty * DimX + k];
		NVal = N[k * DimX + tx];
		Value += MVal * NVal;
	}
	
	P[tid] = Value;
}

void printMatrix(int* mat)
{
	int row = 0; int col = 0;
	for (row = 0; row < LENGTH; row++)
	{
		for (col = 0; col < LENGTH; col++)
		{
			int Destindex = row * LENGTH + col;
			printf( "%2d ", mat[Destindex]);
		}
		printf( "\n");
	}
}

int main()
{
	srand(time(NULL));
	
	const int MatrixSize = LENGTH * LENGTH;
	const int BufferSize = MatrixSize * sizeof(int);
	
	int* M; int* N; int* P;
	
	M = (int*)malloc(BufferSize);
	N = (int*)malloc(BufferSize);
	P = (int*)malloc(BufferSize);
	
	for( int i = 0; i < MatrixSize; i++)
	{
		M[i] = rand()%4; 
		N[i] = rand()%8; 
		P[i] = 0;
	}
	
	int* dev_M; int* dev_N; int* dev_P;
	
	cudaMalloc((void**)&dev_M, BufferSize);
	cudaMalloc((void**)&dev_N, BufferSize);
	cudaMalloc((void**)&dev_P, BufferSize);
	
	cudaMemcpy(dev_M, M, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_N, N, BufferSize, cudaMemcpyHostToDevice);
	
	dim3 Dg(3, 3, 1);
	dim3 Db(4, 4, 1);
	MatrixMulCuda <<<Dg,Db>>> (dev_M, dev_N, dev_P);
	
	cudaMemcpy(P, dev_P, BufferSize, cudaMemcpyDeviceToHost);
	
	printf("\n[matrix M]\n"); printMatrix(M);
	printf("\n[matrix N]\n"); printMatrix(N);
	printf("\n[matrix P]\n"); printMatrix(P);
	
	cudaFree(dev_M); cudaFree(dev_N); cudaFree(dev_P);
	free(M); free(N); free(P);
	
	return 0;
}
