#include <stdio.h>

__global__ void test(int* dev_data1, int* dev_data2, int* dev_data3)
{
	printf("[cuda] dev_data1 addr: %p\n", dev_data1);
	printf("[cuda] dev_data2 addr: %p\n", dev_data2);
	printf("[cuda] dev_data3 addr: %p\n", dev_data3);
}

int main()
{
	int* data1; int* data2; int* data3;
	int* dev_data1; int* dev_data2; int* dev_data3;
	
	data1 = (int*)malloc(sizeof(int));
	data2 = (int*)malloc(sizeof(int));
	data3 = (int*)malloc(sizeof(int));
	
	data1[0] = 1; data2[0] = 2; data3[0] = 3;
	
	printf("[host] data1 addr: %p\n", data1);
	printf("[host] data2 addr: %p\n", data2);
	printf("[host] data3 addr: %p\n", data3);
	
	cudaMalloc((void**)&dev_data1, sizeof(int));
	cudaMalloc((void**)&dev_data2, sizeof(int));
	cudaMalloc((void**)&dev_data3, sizeof(int));
	
	cudaMemcpy(dev_data1, data1, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data2, data2, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data3, data3, sizeof(int), cudaMemcpyHostToDevice);
	
	test<<<1,1>>> (dev_data1, dev_data2, dev_data3);
	
	cudaDeviceSynchronize();
	return 0;
}
