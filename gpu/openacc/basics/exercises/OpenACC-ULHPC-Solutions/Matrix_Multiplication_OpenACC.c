#include<stdio.h>
#include<stdlib.h>
#include <openacc.h>

void Matrix_Multiplication(float *restrict a, float *restrict b, float *restrict c, int width)   
{ 
  float sum = 0;
#pragma acc data copyin(a[0:width*width], b[0:width*width]) copyout(c[0:width*width]) create(sum)
  {
#pragma acc kernels loop collapse(2) reduction(+:sum)
    for(int row = 0; row < width ; ++row)
      {
	for(int col = 0; col < width ; ++col)
	  {
	    sum=0;
	    for(int i = 0; i < width ; ++i)                         
	      {                                                     
		sum += a[row*width+i] * b[i*width+col];      
	      }                                                     
	    c[row*width+col] = sum;                           
	  }
      }
  }
}

int main()
{  
  printf("Programme assumes that matrix size is N*N \n");
  printf("Please enter the N size number \n");
  int N =0;
  scanf("%d", &N);
  
  // Initialize the memory
  float *a, *b, *c;       
  
  // Allocate memory
  a = (float*)malloc(sizeof(float) * (N*N));
  b = (float*)malloc(sizeof(float) * (N*N));
  c = (float*)malloc(sizeof(float) * (N*N));
  
  // Initialize arrays
  for(int i = 0; i < (N*N); i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }
  
  // Fuction call 
  Matrix_Multiplication(a, b, c, N);
  
  // Verification
  for(int i = 0; i < N; i++)
    {
      for(int j = 0; j < N; j++)
	{
          printf("%f ", c[j]);
	}
      printf("\n");
    }
  
  // Deallocate memory
  free(a); 
  free(b); 
  free(c);
  
  return 0;
}
