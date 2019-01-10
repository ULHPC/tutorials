#include <stdio.h>

// This program lists the arguments it has been called with.
//
// Examples of usage:
//
//   ./01-logic_syntax_bugs
//   ./01-logic_syntax_bugs param1
//   ./01-logic_syntax_bugs param1 2
//   ./01-logic_syntax_bugs param1 2 testParam
//


int main(int argc, char** argv)
{
  // number of parameters
  int nb_params = argc - 1;

  // first print a message
  if ( nb_params > 1 )
  {
    printf("This program was called with %i parameters\n", nb_params);
  }
  else if ( nb_params = 1 )
  {
    printf("This program was called with only 1 parameter\n");
  }
  else
  {
    printf("This program was called without any parameter\n");
  }

  // print program name and all parameters
  printf("program = '%s'\n", argv[0] );
  int i;
  for ( i = 0 ; i < nb_params ; i++ )
  {
    printf("parameter %i = '%s'\n", i, argv[i] );
  }

  return 0;
}
