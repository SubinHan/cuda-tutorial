#include<stdio.h>

#define N 10

int main(){
  int a[N], b[N], c[N];

  for(int i=0; i<N; i++){
    a[i] = i;
    b[i] = 100*i;
  }

  for(int i=0; i<N; i++){
    c[i] = a[i] + b[i];
  }

  for(int i=0; i<N; i++){
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  return 1;
}