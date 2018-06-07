#include <stdio.h>
#include <math.h>

// Notes:
//
//  Add -lm on the compilation command line to link with the math library
//  Add -Ddivbyzero -Dinvalidop or -Doverflow to compile only the relevant part


int main(int argc, char** argv)
{

#ifdef divbyzero
  // Division by zero error
  double a = 1.0 / 0.0;
  printf("Division by zero:   1.0 / 0.0 = %e\n", a);
#endif

#ifdef invalidop
  // Invalid operation
  double b = sqrt(-1.0);
  printf("Invalid operation:  sqrt(-1.0) = %e\n", b);
#endif

#ifdef overflow
  // Overflow
  double c = exp( 1e30 );
  printf("Overflow:           exp( 1e30 ) = %e\n", c);
#endif

  return 0;
}
