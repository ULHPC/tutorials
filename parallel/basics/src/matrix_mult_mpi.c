/***********************************************************************************
 * Source: Blaise Barney - https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_mm.c
 * Gist adaptation: https://gist.github.com/kmkurn/39ca673bb37946055b38
 *   Matrix multiply using MPI.
 *   In this code, the master task distributes a matrix multiply operation to
 *   numtasks-1 worker tasks.
 * Compilation:
 * - with 'toolchain/intel': mpiicc matrix_mult_mpi.c -o intel_matrix_mult_mpi
 * - with 'mpi/OpenMPI':     mpicc  matrix_mult_mpi.c -o openmpi_matrix_mult_mpi
 * - with 'mpi/MVAPICH2':    mpicc  matrix_mult_mpi.c -o mvapich2_matrix_mult_mpi
 ***********************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MATSIZE 3000
#define NRA MATSIZE            /* number of rows in matrix A */
#define NCA MATSIZE            /* number of columns in matrix A */
#define NCB MATSIZE            /* number of columns in matrix B */
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[]) {
  int numtasks,            /* number of tasks in partition */
    taskid,                /* a task identifier */
    numworkers,            /* number of worker tasks */
    source,                /* task id of message source */
    dest,                  /* task id of message destination */
    mtype,                 /* message type */
    rows,                  /* rows of matrix A sent to each worker */
    averow, extra, offset, /* used to determine rows sent to each worker */
    i, j, k, rc=1;         /* misc */
  double  a[NRA][NCA],     /* matrix A to be multiplied */
    b[NCA][NCB],           /* matrix B to be multiplied */
    c[NRA][NCB];           /* result matrix C */
  MPI_Status status;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  if (numtasks < 2 ) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }
  numworkers = numtasks-1;


  /**************************** master task ************************************/
  if (taskid == MASTER) {
    printf("matrix_mult_mpi has started with %d tasks.\n",numtasks);
    // printf("Initializing arrays...\n");
    for (i=0; i<NRA; i++)
      for (j=0; j<NCA; j++)
        a[i][j]= i+j;
    for (i=0; i<NCA; i++)
      for (j=0; j<NCB; j++)
        b[i][j]= i*j;

    /* Measure start time */
    double start = MPI_Wtime();

    /* Send matrix data to the worker tasks */
    averow = NRA/numworkers;
    extra = NRA%numworkers;
    offset = 0;
    mtype = FROM_MASTER;
    for (dest=1; dest<=numworkers; dest++)
      {
        rows = (dest <= extra) ? averow+1 : averow;
        // printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
        MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
        MPI_Send(&a[offset][0], rows*NCA, MPI_DOUBLE, dest, mtype,
                 MPI_COMM_WORLD);
        MPI_Send(&b, NCA*NCB, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
        offset = offset + rows;
      }

    /* Receive results from worker tasks */
    mtype = FROM_WORKER;
    for (i=1; i<=numworkers; i++)
      {
        source = i;
        MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&c[offset][0], rows*NCB, MPI_DOUBLE, source, mtype,
                 MPI_COMM_WORLD, &status);
        // printf("Received results from task %d\n",source);
      }

    /* Print results */
    /*
      printf("******************************************************\n");
      printf("Result Matrix:\n");
      for (i=0; i<NRA; i++)
      {
      printf("\n");
      for (j=0; j<NCB; j++)
      printf("%6.2f   ", c[i][j]);
      }
      printf("\n******************************************************\n");
    */

    /* Measure finish time */
    double finish = MPI_Wtime();
    printf("Done in %f seconds.\n", finish - start);
  }


  /**************************** worker task ************************************/
  if (taskid > MASTER) {
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows*NCA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&b, NCA*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

    for (k=0; k<NCB; k++)
      for (i=0; i<rows; i++)
        {
          c[i][k] = 0.0;
          for (j=0; j<NCA; j++)
            c[i][k] = c[i][k] + a[i][j] * b[j][k];
        }
    mtype = FROM_WORKER;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&c, rows*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}
