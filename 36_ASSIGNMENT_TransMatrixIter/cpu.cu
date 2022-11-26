#include <stdio.h>

#define MAX_MATRIX_SIZE 1024
#define SIZE_INC 32

float timevalToFloat(struct timeval* time){
	float val;
	val = time->tv_sec;
	val += (time->tv_usec * 0.000001);
	return val;
}

void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time)
{
	gap_time->tv_sec = end_time->tv_sec - start_time->tv_sec;
	gap_time->tv_usec = end_time->tv_usec - start_time->tv_usec;
	if(gap_time->tv_usec < 0){
		gap_time->tv_usec = gap_time->tv_usec + 1000000;
		gap_time->tv_sec -= 1;
	}
}

void transpose(int* matrix_in, int matrix_size, int* matrix_out)
{
	for(int row = 0; row < matrix_size; row++)
	{
		for(int col = 0; col < matrix_size; col++)
		{
			int offset = row * matrix_size + col;
			int transposed_offset = col * matrix_size + row;
			
			matrix_out[transposed_offset] = matrix_in[offset];
		}
	}
}

void print_matrix(int* matrix, int matrix_size)
{
	for(int row = 0; row < matrix_size; row++)
	{
		for(int col = 0; col < matrix_size; col++)
		{
			int offset = row * matrix_size + col;
			printf("%d ", matrix[offset]);
		}
		printf("\n");
	}
}

void make_matrix(int* matrix, int matrix_size)
{
	int size = matrix_size * matrix_size;
	for(int i = 0; i < size; i++)
	{
		matrix[i] = rand() % 8;
	}
}

int main()
{
	srand(time(NULL));
	
	struct timeval start, end, gap;
	
	for(int current_matrix_size = 32; 
		current_matrix_size <= MAX_MATRIX_SIZE; 
		current_matrix_size += SIZE_INC)
	{
		int* matrix;
		int buffer_size = current_matrix_size * current_matrix_size * sizeof(int);
		
		matrix = (int*) malloc ( buffer_size );
		
		make_matrix(matrix, current_matrix_size);
		
		//printf("origianl: \n");
		//print_matrix(matrix, current_matrix_size);
		
		int* transposed;
		
		transposed = (int*) malloc ( buffer_size );
		
		gettimeofday(&start, NULL);
		transpose(matrix, current_matrix_size, transposed);
		gettimeofday(&end, NULL);
		
		//printf("\ntransposed: \n");
		//print_matrix(transposed, current_matrix_size);
		
		getGapTime(&start, &end, &gap);
		float f_gap = timevalToFloat(&gap);
		printf("[Cuda] LENGTH = %d, time = %.6f\n", current_matrix_size, f_gap);
		
		free(matrix);
		free(transposed);
	}
	
	return 0;
}