#include <stdio.h>
#include "../Common/Timer.h"

#define MATRIX_SIZE 1000

void transpose(int* matrix_in, int* matrix_out)
{
	for(int row = 0; row < MATRIX_SIZE; row++)
	{
		for(int col = 0; col < MATRIX_SIZE; col++)
		{
			int offset = row * MATRIX_SIZE + col;
			int transposed_offset = col * MATRIX_SIZE + row;
			
			matrix_out[transposed_offset] = matrix_in[offset];
		}
	}
}

void print_matrix(int* matrix)
{
	for(int row = 0; row < MATRIX_SIZE; row++)
	{
		for(int col = 0; col < MATRIX_SIZE; col++)
		{
			int offset = row * MATRIX_SIZE + col;
			printf("%d ", matrix[offset]);
		}
		printf("\n");
	}
}

void make_matrix(int* matrix)
{
	int size = MATRIX_SIZE * MATRIX_SIZE;
	for(int i = 0; i < size; i++)
	{
		matrix[i] = rand() % 8;
	}
}

int main()
{
	srand(time(NULL));
	
	int* matrix;
	int buffer_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);
	
	matrix = (int*) malloc ( buffer_size );
	
	make_matrix(matrix);
	
	printf("origianl: \n");
	print_matrix(matrix);
	
	int* transposed;
	
	transposed = (int*) malloc ( buffer_size );
	
	transpose(matrix, transposed);
	
	printf("\ntransposed: \n");
	print_matrix(transposed);
	
	return 0;
}