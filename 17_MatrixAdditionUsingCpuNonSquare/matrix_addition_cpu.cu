#include <stdio.h>
#define LENGTH 12

void MatrixAdd(int* M, int* N, int* P, int width, int height)
{
	int row = 0; 
	int col = 0;
	for (row = 0; row < height; row++)
	{
		for (col = 0; col < width; col++)
		{
			int Destindex = row * width + col;
			P[Destindex] = M[Destindex] + N[Destindex];
		}
	}
}

void printResult(int* M, int* N, int* P, int width, int height)
{
	int row = 0; 
	int col = 0;
	for (row = 0; row < height; row++)
	{
		for (col = 0; col < width; col++)
		{
			int Destindex = row * width + col;
			printf( "%d (= P[%d][%d]) = %d (= M[%d][%d]) + %d (= N[%d][%d]) \n", 
			P[Destindex], row, col, M[Destindex], row, col, N[Destindex], row, col);
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
		M[i] = i;
		N[i] = i;
		P[i] = 0;
	}
	
	MatrixAdd(M, N, P, width, height);
	printResult(M, N, P, width, height);
	
	free(M); free(N); free(P);
	
	return 0;
}
