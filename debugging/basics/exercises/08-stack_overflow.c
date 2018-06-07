#include <stdio.h>
#include <stdlib.h>


int fibo(int n)
{
  if (n == 1)
    return 1;
  else
    return fibo(n-1) + fibo(n-2);
}


int main(int argc, char** argv)
{
  int n = 10;
  int res = fibo(n);
  printf("fibo(%d) = %d\n", n, res);

  return 0;
}
