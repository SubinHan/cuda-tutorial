#include <stdio.h>
#include <math.h>

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

__global__ void transpose(int* matrix_in, int matrix_width, int* matrix_out)
{
	int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
	const int size = matrix_width * matrix_width;
	
	if(thread_index >= size)
		return;

	const int x = thread_index % matrix_width;
	const int y = thread_index / matrix_width;
	
	const int transposed_offset = x * matrix_width + y;
	
	matrix_out[thread_index] = matrix_in[transposed_offset];
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
	
	struct timeval htod_start, htod_end;
	struct timeval gpu_start, gpu_end;
	struct timeval dtoh_start, dtoh_end;
	
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
		
		gettimeofday(&htod_start, NULL);
		cudaMemcpy(d_matrix, matrix, buffer_size, cudaMemcpyHostToDevice);
		gettimeofday(&htod_end, NULL);
		
		struct timeval htod_gap; getGapTime(&htod_start, &htod_end, &htod_gap); 
		float f_htod_gap = timevalToFloat(&htod_gap);
		
		int blocksx = ceil(current_matrix_size * current_matrix_size / 256.0f);
		dim3 threads(256);
		dim3 grid(blocksx);
		
		gettimeofday(&gpu_start, NULL);
		transpose<<<grid, threads>>>(d_matrix, current_matrix_size, d_transposed);
		gettimeofday(&gpu_end, NULL);	
		struct timeval gpu_gap; getGapTime(&gpu_start, &gpu_end, &gpu_gap); 
		float f_gpu_gap = timevalToFloat(&gpu_gap);
		
		int* transposed;
		transposed = (int*) malloc ( buffer_size );
		
		gettimeofday(&dtoh_start, NULL);
		cudaMemcpy(transposed, d_transposed, buffer_size, cudaMemcpyDeviceToHost);
		gettimeofday(&dtoh_end, NULL);
		struct timeval dtoh_gap; getGapTime(&htod_start, &dtoh_end, &dtoh_gap); 
		float f_dtoh_gap = timevalToFloat(&dtoh_gap);
		//printf("origianl: \n");
		//print_matrix(matrix, current_matrix_size);
		//printf("\ntransposed: \n");
		//print_matrix(transposed, current_matrix_size);
		
		float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;
		
		printf("[Cuda] LENGTH = %d, total time = %.6f, htod time = %.6f, GPU time = %.6f, dtoh time = %.6f \n", 
		current_matrix_size, total_gap, f_htod_gap, f_gpu_gap, f_dtoh_gap);
		
		free(matrix);
		free(transposed);
		cudaFree(d_matrix);
		cudaFree(d_transposed);
	}
	
	return 0;
}