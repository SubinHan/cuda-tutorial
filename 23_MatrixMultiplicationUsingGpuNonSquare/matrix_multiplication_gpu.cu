#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_ERR_CHECK(ans) {cudaErrCheck((ans), __FILE__, __LINE__);}
    inline void cudaErrCheck(cudaError_t err, const char *file, int line, bool abort=true){
		printf("333");
		if(err != cudaSuccess){
			fprintf(stderr, "[cuda error] %s (line %d): %s\n", file, line, cudaGetErrorString(err));
			if (abort) exit(err);
		}
	}
 
__global__ void MatrixMulCuda( int*M, int*N, int*P, int M_width, int N_width)
{
	int tid, tx, ty;
	tx = blockDim.x * blockIdx.x + threadIdx.x;
	ty = blockDim.y * blockIdx.y + threadIdx.y;
	
	printf("gridDim.x: %d\n", gridDim.x);
	printf("tx: %d, ty: %d\n", tx, ty);
	
	int DimX = gridDim.x * blockDim.x;
	tid = DimX * ty + tx;
	printf("tid: %d\n", tid);
	
	int Value = 0; int MVal = 0; int NVal = 0;
	for (int k = 0; k < M_width; k++)
	{
		printf("[%d][%d] \n", ty * DimX + k, k * DimX + tx);
	
		MVal = M[ty * M_width + k];
		NVal = N[k * N_width + tx];
		Value += MVal * NVal;
	}
	
	P[tid] = Value;
	
	printf("value: %d", P[tid]);
}

void printMatrix(int* mat, int width, int height)
{
	int row = 0; int col = 0;
	for (row = 0; row < height; row++)
	{
		for (col = 0; col < width; col++)
		{
			int Destindex = row * width + col;
			printf( "%2d ", mat[Destindex]);
		}
		printf( "\n");
	}
}

int main()
{
	srand(time(NULL));
	
	int M_width; int M_height;
	int N_width; int N_height;
	printf("type the M_width: ");
	scanf("%d", &M_width);
	printf("type the M_height: ");
	scanf("%d", &M_height);
	printf("type the N_width: ");
	scanf("%d", &N_width);
	N_height = M_width;
	
	const int MSize = M_width * M_height;
	const int NSize = N_width * N_height;
	const int PSize = M_height * N_width;
	
	const int MBufferSize = MSize * sizeof(int);
	const int NBufferSize = NSize * sizeof(int);
	const int PBufferSize = PSize * sizeof(int);
	
	int* M; int* N; int* P;
	
	M = (int*)malloc(MBufferSize);
	N = (int*)malloc(NBufferSize);
	P = (int*)malloc(PBufferSize);
	
	printf("C %d %d %d", MBufferSize, NBufferSize, PBufferSize);
	
	for(int i = 0; i < MSize; i++)
	{
		M[i] = rand()%4; 
		printf("%d ", M[i]);
	}
	
	for(int i = 0; i < NSize; i++)
	{
		N[i] = rand()%8; 
	}
	
	for(int i = 0; i < PSize; i++)
	{
		P[i] = 0; 
	}
	
	int* dev_M; int* dev_N; int* dev_P;
	
	printf("D");
	
	CUDA_ERR_CHECK(cudaMalloc((void**)&dev_M, MBufferSize));
	
	printf("DA");
	cudaMalloc((void**)&dev_N, NBufferSize);
	cudaMalloc((void**)&dev_P, PBufferSize);
	
	printf("E");
	
	cudaMemcpy(dev_M, M, MBufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_N, N, NBufferSize, cudaMemcpyHostToDevice);
	printf("B");
	
	dim3 Dg(N_width, M_height, 1);
	dim3 Db(1, 1, 1);
	MatrixMulCuda <<<Dg,Db>>> (dev_M, dev_N, dev_P, M_width, N_width);
	
	printf("A");
	
	cudaMemcpy(P, dev_P, PBufferSize, cudaMemcpyDeviceToHost);
	
	printf("\n[matrix M]\n"); printMatrix(M, M_width, M_height);
	printf("\n[matrix N]\n"); printMatrix(N, N_width, N_height);
	printf("\n[matrix P]\n"); printMatrix(P, N_width, M_height);
	
	cudaFree(dev_M); cudaFree(dev_N); cudaFree(dev_P);
	free(M); free(N); free(P);
	
	return 0;
}
