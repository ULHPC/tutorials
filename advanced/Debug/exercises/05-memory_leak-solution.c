#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This program will print all its parameters in uppercase


// Convert buffer to uppercase
// This function allocates a buffer that must be freed by the caller
char* uppercase( char* buffer )
{
  int size = strlen(buffer);
  char* buffer_up = malloc( size+1 );
  int i;
  for ( i = 0 ; i < size ; i++ )
  {
    if ( buffer[i] >= 'a' && buffer[i] <= 'z' )
      buffer_up[i] = buffer[i] + 'A' - 'a';
    else
      buffer_up[i] = buffer[i];
  }
  buffer_up[size] = '\0';

  return buffer_up;
}

int main(int argc, char** argv)
{
  int i;
  for( i = 0 ; i < argc ; i++ )
  {
    char* up_str = uppercase(argv[i]); 
    printf("%s ", up_str );
    free(up_str);
  }
  printf("\n");

  return 0;
}
