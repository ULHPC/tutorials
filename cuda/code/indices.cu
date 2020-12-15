/*
 * FIXME
 */

#include <cstdio>

void printif()
{
  if (threadIdx.x == 1023 && blockIdx.x == 255) {
    printf("Success!\n");
  }
}

int main()
{
  /*
   * Update the execution configuration so that the kernel
   * will print `"Success!"`.
   */

  printif<<<1, 1>>>();
}

