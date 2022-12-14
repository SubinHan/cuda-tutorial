#include <stdio.h>
#define LENGTH 12

void MatrixAdd(int* M, int* N, int* P)
{
	int row = 0; 
	int col = 0;
	for (row = 0; row < LENGTH; row++)
	{
		for (col = 0; col < LENGTH; col++)
		{
			int Destindex = row * LENGTH + col;
			P[Destindex] = M[Destindex] + N[Destindex];
		}
	}
}

void printResult(int* M, int* N, int* P)
{
	int row = 0; 
	int col = 0;
	for (row = 0; row < LENGTH; row++)
	{
		for (col = 0; col < LENGTH; col++)
		{
			int Destindex = row * LENGTH + col;
			printf( "%d (= P[%d][%d]) = %d (= M[%d][%d]) + %d (= N[%d][%d]) \n", 
			P[Destindex], row, col, M[Destindex], row, col, N[Destindex], row, col);
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
		M[i] = i;
		N[i] = i;
		P[i] = 0;
	}
	
	MatrixAdd(M, N, P);
	printResult(M, N, P);
	
	free(M); free(N); free(P);
	
	return 0;
}
