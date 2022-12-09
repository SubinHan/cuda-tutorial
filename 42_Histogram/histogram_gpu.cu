#include <stdio.h>
#include <stdlib.h>
#include "../Common/Timer.h"

#define SIZE (100*1024*1024)

__global__ void HistogramGlobal(unsigned char* Buffer, int* Histogram, int BufferSize)
{
	int Index = blockIdx.x * blockDim.x+ threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	for( ; Index < BufferSize; Index +=offset)
		atomicAdd( &(Histogram[Buffer[Index]]), 1);
}


void* big_random_block(int size) {
	unsigned char *data = (unsigned char*)malloc(size);
	for (int i=0; i<size; i++)
		data[i] = rand();
	return data;
}

int main(void) 
{
	srand(time(NULL));
	struct timeval htod_start, htod_end;
	struct timeval gpu_start, gpu_end;
	struct timeval dtoh_start, dtoh_end;

	const int nBlocks = 512;
	const int nThreads = 512;
	const int Size = SIZE;
	
	unsigned char *host_Buffer = (unsigned char*) big_random_block(Size);
	
	int* host_Histogram;
	cudaMallocHost((void**)&host_Histogram,256*sizeof(int));
	
	unsigned char* dev_Buffer;
	int* dev_Histogram;
	
	cudaMalloc((void**)&dev_Buffer, Size);
	cudaMalloc((void**)&dev_Histogram, 256*sizeof(int));
	
	cudaMemset(dev_Buffer, 0, Size);
	cudaMemset(dev_Histogram, 0, 256*sizeof(int));
	
	gettimeofday(&htod_start, NULL);
	cudaMemcpy(dev_Buffer, host_Buffer, Size, cudaMemcpyHostToDevice);
	gettimeofday(&htod_end, NULL);
	
	gettimeofday(&gpu_start, NULL);
	HistogramGlobal<<<nBlocks,nThreads>>>(dev_Buffer,dev_Histogram,Size);
	cudaDeviceSynchronize();
	gettimeofday(&gpu_end, NULL);
	
	gettimeofday(&dtoh_start, NULL);
	cudaMemcpy(host_Histogram, dev_Histogram, 256*sizeof(int), cudaMemcpyDeviceToHost);
	gettimeofday(&dtoh_end, NULL);
	
	struct timeval dtoh_gap, gpu_gap, htod_gap;
	getGapTime(&dtoh_start, &dtoh_end, &dtoh_gap);
	getGapTime(&gpu_start, &gpu_end, &gpu_gap);
	getGapTime(&htod_start, &htod_end, &htod_gap);
	
	float f_htod_gap = timevalToFloat(&htod_gap);
	float f_gpu_gap = timevalToFloat(&gpu_gap);
	float f_dtoh_gap = timevalToFloat(&dtoh_gap);
	float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;
	
	for(int i = 0; i < 10; i++)
	{
		printf("Histogram[%d] : %d\n",i,host_Histogram[i]);
	}
	printf(" ......\n");
	for(int i = 250; i < 256; i++)
	{
		printf("Histogram[%d] : %d\n",i,host_Histogram[i]);
	}
	printf("[Cuda] total time = %.6f, htod time = %.6f, GPU time = %.6f, dtoh time = %.6f \n",
	total_gap, f_htod_gap, f_gpu_gap, f_dtoh_gap);
	
	cudaFree(dev_Buffer);
	cudaFree(dev_Histogram);
	return 0;
}
