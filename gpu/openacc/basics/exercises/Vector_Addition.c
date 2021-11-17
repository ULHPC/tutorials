// Author:Ezhilmathi Krishnasamy (ezhilmathi.krishnasamy@uni.lu)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define MAX_ERR 1e-6

// function that adds two vector 
void Vector_Addition(float *a, float *b, float *c, int n) 
{
  for(int i = 0; i < n; i ++)
    {
      c[i] = a[i] + b[i];
    }
}

int main()
{
  printf("This program does the addition of two vectors \n");
  printf("Please specify the vector size = ");
  int N;
  scanf("%d",&N);

  // Initialize the memory on the host
  float *a, *b, *c;       
  
  // Allocate host memory
  a = (float*)malloc(sizeof(float) * N);
  b = (float*)malloc(sizeof(float) * N);
  c = (float*)malloc(sizeof(float) * N);
  
  // Initialize host arrays
  for(int i = 0; i < N; i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }
    
  // Executing Vector Addition funtion 
  Vector_Addition(a, b, c, N);
  
  // Verification
  for(int i = 0; i < N; i++)
    {
      assert(fabs(c[i] - (a[i] + b[i])) < MAX_ERR);
    }
  printf("PASSED\n");
    
  // Deallocate host memory
  free(a); 
  free(b); 
  free(c);

  return 0;
}
