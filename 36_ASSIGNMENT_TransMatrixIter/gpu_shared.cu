#include <stdio.h>
#include <math.h>

#define MAX_MATRIX_SIZE 1024
#define SIZE_INC 32
#define MAX_THREAD_SIZE 256

__global__ void transpose(int* matrix_in, int matrix_width, int* matrix_out)
{
	int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
	const int size = matrix_width * matrix_width;
	
	if(thread_index >= size)
		return;

	const int x = thread_index % matrix_width;
	const int y = thread_index / matrix_width;
	
	const int transposed_offset = x * matrix_width + y;
	
	__shared__ int shared_matrix[MAX_THREAD_SIZE];
	shared_matrix[threadIdx.x] = matrix_in[transposed_offset];
	
	__syncthreads();
	
	matrix_out[thread_index] = shared_matrix[threadIdx.x];
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
	
	for(int current_matrix_size = 32; 
		current_matrix_size <= MAX_MATRIX_SIZE; 
		current_matrix_size += SIZE_INC)
	{
		int* matrix;
		int buffer_size = current_matrix_size * current_matrix_size * sizeof(int);
		
		matrix = (int*) malloc ( buffer_size );
		make_matrix(matrix, current_matrix_size);
		
		int* d_matrix;
		int* d_transposed;
		
		cudaMalloc((void**)&d_matrix, buffer_size);
		cudaMalloc((void**)&d_transposed, buffer_size);
		
		cudaMemcpy(d_matrix, matrix, buffer_size, cudaMemcpyHostToDevice);
		
		int blocksx = ceil(current_matrix_size * current_matrix_size / (float)MAX_THREAD_SIZE);
		dim3 threads(MAX_THREAD_SIZE);
		dim3 grid(blocksx);
		
		transpose<<<grid, threads>>>(d_matrix, current_matrix_size, d_transposed);	
		
		int* transposed;
		transposed = (int*) malloc ( buffer_size );
		
		cudaMemcpy(transposed, d_transposed, buffer_size, cudaMemcpyDeviceToHost);
		
		//printf("origianl: \n");
		//print_matrix(matrix, current_matrix_size);
		//printf("\ntransposed: \n");
		//print_matrix(transposed, current_matrix_size);
		
		free(matrix);
		free(transposed);
		cudaFree(d_matrix);
		cudaFree(d_transposed);
	}
	return 0;
}