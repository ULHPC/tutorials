/* FIXME
 * Correct, and refactor 'loop' to be a CUDA Kernel.
 * The new kernel should only do the work
 * of 1 iteration of the original loop.
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
   * Use 1 block of threads.
   */

  int N = 10;
  loop(N);
}

