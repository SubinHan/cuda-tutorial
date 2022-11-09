#include <stdio.h>
#include <stdlib.h>
#include "../Common/Timer.h"

#define LENGTH 1024

__global__ void MatrixMulCuda( int*M, int*N, int*P )
{
	int tid, tx, ty;
	tx = blockDim.x * blockIdx.x + threadIdx.x;
	ty = blockDim.y * blockIdx.y + threadIdx.y;
	tid = LENGTH * ty + tx;
	
	int Value = 0; int MVal = 0; int NVal = 0;
	
	for (int k = 0; k < LENGTH; k++)
	{
		MVal = M[ty * LENGTH + k];
		NVal = N[k * LENGTH + tx];
		Value += MVal * NVal;
	}
	
	P[tid] = Value;
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
	
	struct timeval htod_start, htod_end; 
	struct timeval gpu_start, gpu_end; 
	struct timeval dtoh_start, dtoh_end;
	
	gettimeofday(&htod_start, NULL);
	cudaMemcpy(dev_M, M, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_N, N, BufferSize, cudaMemcpyHostToDevice);
	gettimeofday(&htod_end, NULL);
	struct timeval htod_gap; getGapTime(&htod_start, &htod_end, &htod_gap); 
	float f_htod_gap = timevalToFloat(&htod_gap);
	
	dim3 Dg(32, 1, 1); dim3 Db(32, 1, 1);
	
	gettimeofday(&gpu_start, NULL);
	MatrixMulCuda <<<Dg,Db>>> (dev_M, dev_N, dev_P);
	cudaDeviceSynchronize();
	gettimeofday(&gpu_end, NULL);
	struct timeval gpu_gap; getGapTime(&gpu_start, &gpu_end, &gpu_gap); 
	float f_gpu_gap = timevalToFloat(&gpu_gap);
	
	gettimeofday(&dtoh_start, NULL);
	cudaMemcpy(P, dev_P, BufferSize, cudaMemcpyDeviceToHost);
	gettimeofday(&dtoh_end, NULL);
	struct timeval dtoh_gap; getGapTime(&htod_start, &dtoh_end, &dtoh_gap); 
	float f_dtoh_gap = timevalToFloat(&dtoh_gap);
	
	float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;
	
	printf("[Cuda] LENGTH = %d, total time = %.6f, htod time = %.6f, GPU time = %.6f, dtoh time = %.6f \n", 
	LENGTH, total_gap, f_htod_gap, f_gpu_gap, f_dtoh_gap);
	
	cudaFree(dev_M); cudaFree(dev_N); cudaFree(dev_P);
	free(M); free(N); free(P);
	
	return 0;
}
