/***********************************************************************************
 * Source: Blaise Barney - https://computing.llnl.gov/tutorials/openMP/exercise.html
 * Compilation:
 * - with 'toolchain/intel': icc -qopenmp hello_openmp.c -o hello_openmp
 * - with 'toolchain/foss':  gcc -fopenmp hello_openmp.c -o hello_openmp
 ***********************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])  {
  int nthreads, tid;

  // Fork a team of threads giving them their own copies of variables
#pragma omp parallel private(nthreads, tid)
  {
    tid = omp_get_thread_num();                    // Obtain thread number
    printf("Hello World from thread = %d\n", tid);

    // Only master thread does this
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
  }  // All threads join master thread and disband
  return 0;
}
