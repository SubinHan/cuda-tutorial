#include <stdio.h>
#include <stdlib.h>

__global__ void atomicShared(int* nThreadCount)
{
	__shared__ int nCount;
	
	if(threadIdx.x == 0)
		nCount = 0;
	__syncthreads();
	
	atomicAdd(&nCount, 1);
	__syncthreads();

	if(threadIdx.x == 0)
		atomicAdd(nThreadCount, nCount);
}

int main()
{
	const int nBlocks = 10000; const int nThreads = 1024;
	int nThreadCount = 0; int* dev_nThreadCount;
	
	cudaMalloc((void**)&dev_nThreadCount, sizeof(int));
	
	cudaMemset(dev_nThreadCount, 0, sizeof(int));
	
	atomicShared<<<nBlocks,nThreads>>>(dev_nThreadCount);
	
	cudaMemcpy(&nThreadCount, dev_nThreadCount, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("스레드 실행 개수: %d 개 \n",nThreadCount);
	
	cudaFree(dev_nThreadCount);
	
	return 0;
}
