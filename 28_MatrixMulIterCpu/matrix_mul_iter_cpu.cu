#include <stdio.h>
#include "../Common/Timer.h"

void MatrixMul(int* M, int* N, int* P, int LENGTH)
{
	int row = 0; int col = 0;
	for (row = 0; row < LENGTH; row++)
	{
		for (col = 0; col < LENGTH; col++)
		{
			int Destindex = row * LENGTH + col;
			for (int k= 0; k < LENGTH; k++)
			{
				P[Destindex] += M[row * LENGTH + k] * N[col + k * LENGTH];
			}
		}
	}
}

int main()
{
	srand(time(NULL));
	struct timeval start, end, gap;
	
	int MatrixWidth; int MatrixHeight; int MatrixSize; int BufferSize;
	int* M; int* N; int* P;
	
	for(int LENGTH=8; LENGTH<1025; LENGTH+=8)
	{ 
		MatrixWidth = LENGTH;
		MatrixHeight = LENGTH;
		MatrixSize = MatrixWidth * MatrixHeight;
		BufferSize = MatrixSize * sizeof(int);
		
		M = (int*)malloc(BufferSize);
		N = (int*)malloc(BufferSize);
		P = (int*)malloc(BufferSize);
		
		for(int i = 0; i < MatrixSize; i++)
		{
			M[i] = rand()%4; N[i] = rand()%8; P[i] = 0;
		}
		gettimeofday(&start, NULL);
		MatrixMul(M, N, P, LENGTH);
		gettimeofday(&end, NULL);
		
		getGapTime(&start, &end, &gap);
		float f_gap = timevalToFloat(&gap);
		
		printf("LENGTH = %d, CPU time = %.6f \n", LENGTH, f_gap);
		
		free(M); free(N); free(P);
	} 
	return 0;
}
