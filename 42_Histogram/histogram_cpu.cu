#include <stdio.h>
#include <stdlib.h>
#include "../Common/Timer.h"

#define SIZE (100*1024*1024)

void* big_random_block(int size) {
	unsigned char *data = (unsigned char*)malloc(size);
	for (int i=0; i<size; i++)
		data[i] = rand();
	return data;
}

int main(void) 
{
	srand(time(NULL));
	
	struct timeval start, end, gap;
	
	unsigned char *buffer = (unsigned char*) big_random_block(SIZE);
	
	unsigned int histo[256];
	
	for (int i=0; i<256; i++)
		histo[i] = 0;
		
	gettimeofday(&start, NULL);
	for (int i=0; i<SIZE; i++)
		histo[buffer[i]]++;
	gettimeofday(&end, NULL);
	
	getGapTime(&start, &end, &gap);
	float f_gap = timevalToFloat(&gap);
	printf("[CPU] Time to generate: %.6f seconds \n", f_gap);
	
	long histoCount = 0;
	
	for (int i=0; i<256; i++)
		histoCount += histo[i];
		
	printf("Histogram Sum: %ld\n", histoCount);
	
	free(buffer);
	return 0;
}
