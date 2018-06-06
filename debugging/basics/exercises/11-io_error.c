#include <stdio.h>

#define SIZE 1024

int main(int argc, char** argv)
{
  FILE* file = fopen("/foo.bar", "r");


  char buffer[SIZE];
  fscanf( file, "%s", buffer );
  printf("'%s'\n", buffer);

  return 0;
}
