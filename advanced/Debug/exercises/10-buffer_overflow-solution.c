#include <stdio.h>
#include <string.h>

#define SIZE 10

void print_uppercase(char* str)
{
  // buffer to store uppercase string
  char str_up[SIZE];
  int i = 0;

  // firt copy string
  // use strncpy() to avoid buffer overflow
  strncpy( str_up, str, SIZE-1 );
  str_up[SIZE-1] = '\0';

  // change lowercase to uppercase
  for ( ; i < SIZE ; i++ )
  {
    if ( str_up[i] >= 'a' && str_up[i] <= 'z' )
      str_up[i] = str_up[i] + 'A' - 'a';
  }

  // print
  printf("'%s' -> '%s'\n", str, str_up);
}


int main(int argc, char** argv)
{
  print_uppercase("First test");
  print_uppercase("This is the second test!");

  return 0;
}
