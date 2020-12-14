/* FIXME
 * Fix and refactor 'loop' to be a CUDA Kernel, launched with 2 or more blocks
 * The new kernel should only do the work of 1 iteration of the original loop.
 */

#include <cstdio>

void loop(int N)
{
  for (int i = 0; i < N; ++i) {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{
  /*
   * When refactoring 'loop' to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, be sure to use more than 1 block in
   * the execution configuration.
   */

  int N = 10;
  loop(N);
}

