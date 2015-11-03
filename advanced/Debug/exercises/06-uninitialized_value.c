#include <stdio.h>
#include <stdlib.h>



int main(int argc, char** argv)
{

#ifdef uninitstatic
  double x,y;
  x = y + 2;
  printf("x = %f\n",x);
  printf("y = %f\n",y);

#endif

#ifdef uninitdynamic
  int size = 10;
  double *array = malloc( sizeof(double) * size );

  int i;
  for( i = 1 ; i < size ; i++ )
    array[i] = array[i-1];

  for( i = 0 ; i < size ; i++ )
    printf(" array[%i] = %f\n", i, array[i] );

  free(array);
#endif

#ifdef uninitnonalloc
  int size = 10;
  double *array1, *array2 = malloc( sizeof(double) * size );

  int i;
  for( i = 0 ; i < size ; i++ )
    array2[i] = array1[i];

  for( i = 0 ; i < size ; i++ )
    printf(" array2[%i] = %f\n", i, array2[i] );

  free(array2);
#endif

 return 0;
}
