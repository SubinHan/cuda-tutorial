#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LENGTH 1000

void MatrixMul(int* M, int* N, int* P)
{
	int row = 0; int col = 0;
	
	for (row = 0; row < LENGTH; row++)
	{
		for (col = 0; col < LENGTH; col++)
		{
			int Destindex = row * LENGTH + col;
			for (int k = 0; k < LENGTH; k++)
			{
				P[Destindex] += M[row * LENGTH + k] * N[col + k * LENGTH];
			}
		}
	}
}

void printResult(int* M, int* N, int* P)
{
	int row = 0; int col = 0;
	for (row = 0; row < LENGTH; row++)
	{
		for (col = 0; col < LENGTH; col++)
		{
			int Destindex = row * LENGTH + col;
			for (int k=0; k < LENGTH; k++)
			{
				printf("(%d = M[%d][%d], %d = N[%d][%d]) \n", 
				M[row * LENGTH + k], row, k, N[col + k * LENGTH], k, col);
			}
			printf( "%d, P[%d][%d] = M[%d][.] dot N[.][%d] \n\n", 
			P[Destindex], row, col, row, col );
		}
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
	
	time_t start, end;
	time(&start);
	MatrixMul(M, N, P);
	time(&end);
	//printResult(M, N, P);
	
	printf("%lld", end - start);
	
	free(M); free(N); free(P);
	return 0;
}
