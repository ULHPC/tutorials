#include <stdio.h>
#include <stdlib.h>

// Notes:
//
//  Add -Duninitstatic -Duninitdynamic or -Duninitnonalloc to compile only the relevant part



int main(int argc, char** argv)
{

#ifdef uninitstatic
  double x,y;
  y = 0;
  x = y + 2;
  printf("x = %f\n",x);
  printf("y = %f\n",y);

#endif

#ifdef uninitdynamic
  int size = 10;
  double *array = malloc( sizeof(double) * size );

  array[0] = 42.0;
  int i;
  for( i = 1 ; i < size ; i++ )
    array[i] = array[i-1];

  for( i = 0 ; i < size ; i++ )
    printf(" array[%i] = %f\n", i, array[i] );

  free(array);
#endif

#ifdef uninitnonalloc
  int size = 10;
  double *array1 = malloc( sizeof(double) * size );
  double *array2 = malloc( sizeof(double) * size );

  int i;
  // initialize array1
  for( i = 0 ; i < size ; i++ )
    array1[i] = 0.0;
  
  for( i = 0 ; i < size ; i++ )
    array2[i] = array1[i];

  for( i = 0 ; i < size ; i++ )
    printf(" array2[%i] = %f\n", i, array2[i] );

  free(array1);
  free(array2);
#endif

 return 0;
}
