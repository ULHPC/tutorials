#include <stdio.h>
#include <stdlib.h>

// This program computes factorial.
//
// Examples of usage:
//
//   ./02-integer_overflow 4
//   ./02-integer_overflow 10
//   ./02-integer_overflow 20
//


int factorial(int n)
{
  int result = 1;
  int i;
  for ( i = 2 ; i <= n ; i++ )
  {
    result *= i;
  }
  return result;
}

int main(int argc, char** argv)
{
  // check number of parameters
  if ( argc != 2)
  {
    printf("Error: exactly one parameter is required!\n");
    return 1;
  }

  // get first parameter
  int n = atoi(argv[1]);
  int fact = factorial(n);
  printf(" fact(%i) = %i\n", n, fact);

  return 0;
}
