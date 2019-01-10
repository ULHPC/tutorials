/***********************************************************************************
 * @file   hello_mpi.c
 * @author Sebastien Varrette <Sebastien.Varrette@uni.lu>
 * @date   Tue Nov 27 2012
 * Compilation:
 * - with 'toolchain/intel':  mpiicc hello_mpi.c -o intel_hello_mpi
 * - with 'mpi/OpenMPI':      mpicc  hello_mpi.c -o openmpi_hello_mpi
 * - with 'mpi/MVAPICH2':     mpicc  hello_mpi.c -o mvapich2_hello_mpi
 ***********************************************************************************/
#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit  */
#include <stdarg.h>    /* for va_{list,args... */
#include <unistd.h>    /* for sleep */
#include <mpi.h>

int id = 0; // MPI id for the current process (set global to be used in xprintf)

/**
 * Redefinition of the printf to include the buffer flushing
 */
void xprintf(char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("[Node %i] ", id);
    vprintf(format, args);
    fflush(stdout);
}

int main(int argc, char *argv[]) {
    int p;              // MPI specific: number of processors
    double elapsed_time = 0.0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&p);  // Get #processes
    MPI_Comm_rank(MPI_COMM_WORLD,&id); // Get current rank

    // Get the name of the processor
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (id == 0) {
        xprintf("Total Number of processes : %i\n",p);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();
    // Let's go -- do something
    xprintf("Hello world from processor %s, rank %d out of %d processors\n",
            processor_name, id, p);

    // at the end, compute global elapsed time
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();
    if (id == 0) {
      xprintf("Global Elapsed time: %2f s\n", elapsed_time);
    }
    MPI_Finalize();
    return 0;
}
