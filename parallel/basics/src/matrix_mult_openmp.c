/***********************************************************************************
 * Source: adapted from Blaise Barney -
 *         https://computing.llnl.gov/tutorials/openMP/exercise.html
 *   Matrix multiply using OpenMP. Threads share row iterations according to a
 *   predefined chunk size.
 * Compilation:
 * - with 'toolchain/intel': icc -qopenmp matrix_mult_openmp.c -o matrix_mult_openmp
 * - with 'toolchain/foss':  gcc -fopenmp matrix_mult_openmp.c -o matrix_mult_openmp
 ***********************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MATSIZE 2000
#define NRA MATSIZE        /* number of rows in matrix A */
#define NCA MATSIZE        /* number of columns in matrix A */
#define NCB MATSIZE        /* number of columns in matrix B */

int main (int argc, char *argv[]) {
  int tid, nthreads, i, j, k, chunk;
  double elapsed_time = 0.0;
  double  a[NRA][NCA],     /* matrix A to be multiplied */
    b[NCA][NCB],           /* matrix B to be multiplied */
    c[NRA][NCB];           /* result matrix C */

  chunk = 10;              /* set loop iteration chunk size */
  elapsed_time = -omp_get_wtime();

  /*** Spawn a parallel region explicitly scoping all variables ***/
#pragma omp parallel shared(a,b,c,nthreads,chunk) private(tid,i,j,k)
  {
    tid = omp_get_thread_num();
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("Starting matrix multiple example with %d threads\n",nthreads);
      printf("Initializing matrices...\n");
    }
    /*** Initialize matrices ***/
#pragma omp for schedule (static, chunk)
    for (i=0; i<NRA; i++)
      for (j=0; j<NCA; j++)
        a[i][j]= i+j;
#pragma omp for schedule (static, chunk)
    for (i=0; i<NCA; i++)
      for (j=0; j<NCB; j++)
        b[i][j]= i*j;
#pragma omp for schedule (static, chunk)
    for (i=0; i<NRA; i++)
      for (j=0; j<NCB; j++)
        c[i][j]= 0;

    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
    printf("Thread %d starting matrix multiply...\n",tid);
#pragma omp for schedule (static, chunk)
    for (i=0; i<NRA; i++)
      {
        if (MATSIZE <= 100)
          printf("Thread=%d did row=%d\n",tid,i);
        for(j=0; j<NCB; j++)
          for (k=0; k<NCA; k++)
            c[i][j] += a[i][k] * b[k][j];
      }
  }   /*** End of parallel region ***/
  elapsed_time += omp_get_wtime();

  if (MATSIZE <= 100) {
    /*** Print results ***/
    printf("******************************************************\n");
    printf("Result Matrix:\n");
    for (i=0; i<NRA; i++)
      {
        for (j=0; j<NCB; j++)
          printf("%6.2f   ", c[i][j]);
        printf("\n");
      }
    printf("******************************************************\n");
  }
  printf("Global Elapsed time: %2f s\n", elapsed_time);
  //printf ("Done.\n");
  return 0;
}
