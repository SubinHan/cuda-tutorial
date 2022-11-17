/*
	nvcc는 C++기반으로 작성되어 있으므로..
	c 기반의 코드를 include 할 때에는 extern "C" 가드를 작성해주어야 함!
	이를 작성하지 않을 시 LINK 에러가 나니 주의할 것.
*/

#include <stdio.h>
#include "../Common/Timer.h"
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

void printTime(struct timeval time)
{
  printf("%ld.%ld\n", time.tv_sec, time.tv_usec);
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
	
	struct timeval start, end, diff;
  gettimeofday(&start, NULL);
	MatrixMul(M, N, P);
  gettimeofday(&end, NULL);
  diff.tv_sec = end.tv_sec - start.tv_sec;
  diff.tv_usec = end.tv_usec - start.tv_usec;
  
  if(diff.tv_usec < 0l)
  {
	diff.tv_sec--;
	diff.tv_usec += 1000000;
  }
  
  printTime(diff);
  printTime(start);
  printTime(end);
	//printResult(M, N, P);
	
	free(M); free(N); free(P);
	return 0;
}