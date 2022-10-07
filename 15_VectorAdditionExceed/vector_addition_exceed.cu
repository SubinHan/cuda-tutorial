#include <stdio.h>

__global__ void device_add(int *a, int *b, int *c, int cnt)
{
	int capacity = 512 * 65535;
	int count = 0;
	while(count < cnt / capacity)
	{
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		int newIndex = index + (capacity * count);
		
		//printf("%d, %d\n", index, count);
		c[newIndex] = a[newIndex] + b[newIndex];
		//printf("%d + %d = %d\n", a[newIndex], b[newIndex], c[newIndex]);
		count++;
		
		
		//if(newIndex > capacity * 2)
		//	printf("%d, %d\n", newIndex, c[newIndex]);
	}
}

int main()
{
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;
	int arr_cnt = 33553920 * 2;
	
	// allocate the memory on the CPU
	a = (int*)malloc( arr_cnt * sizeof(int) );
	b = (int*)malloc( arr_cnt * sizeof(int) );
	c = (int*)malloc( arr_cnt * sizeof(int) );
	
	// allocate the memory on the GPU
	cudaMalloc( (void**)&dev_a, arr_cnt * sizeof(int) );
	cudaMalloc( (void**)&dev_b, arr_cnt * sizeof(int) );
	cudaMalloc( (void**)&dev_c, arr_cnt * sizeof(int) );
	
	// fill the arrays 'a' and 'b' on the CPU
	for (int i=0; i<arr_cnt; i++) {
		a[i] = i; b[i] = i;
	}

	// copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy( dev_a, a, arr_cnt * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, b, arr_cnt * sizeof(int), cudaMemcpyHostToDevice );
	
	int threads_per_block = 512;
	int blocks = 65535;
	
	device_add <<< blocks, threads_per_block>>> ( dev_a, dev_b, dev_c, arr_cnt );
	
	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy( c, dev_c, arr_cnt * sizeof(int), cudaMemcpyDeviceToHost );
	
	printf("%d", arr_cnt);
	
	// verify that the GPU did the work we requested
	bool success = true;
	for (int i=0; i<arr_cnt; i++) {
		if ((a[i] + b[i]) != c[i]) {
			printf( "Error: %d + %d != %d\n", a[i], b[i], c[i] );
			
			success = false;
			break;
		}
	}
	if (success)
		printf( "We did it!\n" );
	else 
		printf( "We failed\n" );
		
	// free the memory we allocated on the GPU
	cudaFree( dev_a ); cudaFree( dev_b ); cudaFree( dev_c );
	
	// free the memory we allocated on the CPU
	free( a ); free( b ); free( c );
	
	return 0;
}

