#include <stdio.h>
#include <omp.h>
#include "mpi.h"

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    #pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        printf("Hello from thread %d and process %d on processor %s\n",
               thread_num, rank, processor_name);
    }

    MPI_Finalize();
}
