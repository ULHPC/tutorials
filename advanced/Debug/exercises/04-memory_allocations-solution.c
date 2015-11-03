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
  // We need to check the return value of malloc
  if ( array == NULL )
  {
    printf("Error: failed to allocate memory with malloc()\n");
    return 1;
  }
  array[0] = 2.0;
  free(array);
#endif

#ifdef doublefree
  int *p = malloc( 2 * sizeof(int) );
  // We need to check the return value of malloc
  if ( p == NULL )
  {
    printf("Error: failed to allocate memory with malloc()\n");
    return 1;
  }
  free(p);
  // p has already been freed
  // free(p);  
#endif

#ifdef freenonalloc
  double d[100];
  // d is a static variable not alllocated with malloc()
  // free(d);
#endif

#ifdef doublealloc
  void *p;
  p = malloc( 100 );
  // We need to check the return value of malloc
  if ( p == NULL )
  {
    printf("Error: failed to allocate memory with malloc()\n");
    return 1;
  }
  // if we don't free p now, the memory will be lost when p is reassigned
  free(p);
      
  p = malloc( 10 );
  if ( p == NULL )
  {
    printf("Error: failed to allocate memory with malloc()\n");
    return 1;
  }
  free(p);
#endif

  return 0;
}
