#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void MatrixMul(int* M, int* N, int* P, int M_width, int M_height, int N_width)
{
	int row = 0; int col = 0;
	
	for (row = 0; row < M_height; row++)
	{
		for (col = 0; col < N_width; col++)
		{
			int Destindex = row * N_width + col;
			for (int k = 0; k < M_width; k++)
			{
				P[Destindex] += M[row * M_width + k] * N[col + k * N_width];
			}
		}
	}
}

void printResult(int* M, int* N, int* P, int M_width, int M_height, int N_width)
{
	int row = 0; int col = 0;
	for (row = 0; row < M_height; row++)
	{
		for (col = 0; col < N_width; col++)
		{
			int Destindex = row * N_width + col;
			for (int k=0; k < M_width; k++)
			{
				printf("(%d = M[%d][%d], %d = N[%d][%d]) \n", 
				M[row * M_width + k], row, k, N[col + k * N_width], k, col);
			}
			printf( "%d, P[%d][%d] = M[%d][.] dot N[.][%d] \n\n", 
			P[Destindex], row, col, row, col );
		}
	}
}

void printMatrix(int *M, int width, int height)
{
	int row = 0; int col = 0;
	for (row = 0; row < height; row++)
	{
		for (col = 0; col < width; col++)
		{
			printf("[%2d]", M[width * row + col]);
		}
		printf("\n");
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
	
	for(int i = 0; i < MBufferSize; i++)
	{
		M[i] = rand()%4; 
	}
	
	for(int i = 0; i < NBufferSize; i++)
	{
		N[i] = rand()%8; 
	}
	
	for(int i = 0; i < PBufferSize; i++)
	{
		P[i] = 0; 
	}
	
	MatrixMul(M, N, P, M_width, M_height, N_width);
	
	printResult(M, N, P, M_width, M_height, N_width);
	
	printMatrix(M, M_width, M_height);
	printf("*\n");
	printMatrix(N, N_width, M_width);
	printf("=\n");
	printMatrix(P, N_width, M_height);
	
	free(M); free(N); free(P);
	return 0;
}
