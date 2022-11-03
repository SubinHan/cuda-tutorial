#include <stdio.h>
#include <time.h>
#include <sys/time.h>

int main()
{
	time_t timer;
	time(&timer);
	char* curTime = ctime(&timer);
	
	printf("1970년 1월 1일 0시 이후로 %lld 초가 지났습니다. \n", timer);
	printf("현재 시간은 \n%s 입니다. \n", curTime);
	
	struct timeval utimer;
	gettimeofday(&utimer, NULL);
	printf("%lld seconds + %lld microseconds\n", utimer.tv_sec, utimer.tv_usec);
}