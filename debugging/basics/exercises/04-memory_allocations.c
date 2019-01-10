#include <stdio.h>
#include <stdlib.h>

// Notes:
//
//  Add -Dfailedalloc -Ddoublefree -Dfreenonalloc or -Ddoublealloc to compile only the relevant part


int main(int argc, char** argv)
{

#ifdef failedalloc
  short SIZE = 1111;
  double *array = malloc( sizeof(double) * -SIZE );
  array[0] = 2.0;
  free(array);
#endif

#ifdef doublefree
  int *p = malloc( 2 * sizeof(int) );
  free(p);
  free(p);
#endif

#ifdef freenonalloc
  double d[100];
  free(d);
#endif

#ifdef doublealloc
  void *p;
  p = malloc( 100 );
  p = malloc( 10 );
  free(p);
#endif

  return 0;
}
