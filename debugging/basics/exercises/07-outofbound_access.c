#include <stdio.h>
#include <stdlib.h>



int main(int argc, char** argv)
{
  int size = 100;

  double *fibo = malloc( sizeof(double) * size );
  fibo[0] = 1; fibo[1] = 1;

  int i;
  for( i = 1 ; i < size ; i++ )
  {
    fibo[i] = fibo[i-1] + fibo[i-2];
  }

  printf(" fibo = %f\n", fibo[size] );

  free(fibo);

 return 0;
}
