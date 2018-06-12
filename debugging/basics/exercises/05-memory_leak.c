#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This program will print all its parameters in uppercase


// Convert buffer to uppercase
char* uppercase( char* buffer )
{
  int size = strlen(buffer);
  char* buffer_up = malloc( size );
  int i;
  for ( i = 0 ; i < size ; i++ )
  {
    if ( buffer[i] >= 'a' && buffer[i] <= 'z' )
      buffer_up[i] = buffer[i] + 'A' - 'a';
    else
      buffer_up[i] = buffer[i];
  }

  return buffer_up;
}

int main(int argc, char** argv)
{
  int i;
  for( i = 0 ; i < argc ; i++ )
  {
    printf("%s ", uppercase(argv[i]) );
  }
  printf("\n");

  return 0;
}
