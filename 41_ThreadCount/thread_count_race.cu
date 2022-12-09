#include <stdio.h>
#include <stdlib.h>

__global__ void threadCountDataRace(int* nThreadCount)
{
	(*nThreadCount)++;
}
__global__ void atomicGlobal(int* nThreadCount)
{
	atomicAdd(nThreadCount, 1);
}

int main()
{
	const int nBlocks = 10000; const int nThreads = 1024;
	int nThreadCount = 0; int* dev_nThreadCount;
	
	cudaMalloc((void**)&dev_nThreadCount, sizeof(int));
	
	cudaMemset(dev_nThreadCount, 0, sizeof(int));
	
	threadCountDataRace<<<nBlocks,nThreads>>>(dev_nThreadCount);
	
	cudaMemcpy(&nThreadCount, dev_nThreadCount, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("스레드 실행 개수: %d 개 \n",nThreadCount);
	
	cudaFree(dev_nThreadCount);
	
	return 0;
}
