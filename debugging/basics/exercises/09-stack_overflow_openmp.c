#include <stdio.h>

// This example uses OpenMP. It must be compiled with the -fopenmp option.

int main(int argc, char** argv)
{
  double U1[1000000];

  #pragma omp parallel default(none) private(U1)
  {
    U1[0]=1.0;
    printf("Hello World: %f\n", U1[0]);
  }

  return 0;
}
