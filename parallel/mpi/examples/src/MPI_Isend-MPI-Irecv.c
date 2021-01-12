#include <stdio.h>
#include "mpi.h"


////////////
//MPI_Isend
////////////
//
// int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
//              MPI_Comm comm, MPI_Request *request)
//
// This example uses MPI_Isend to do a non-blocking send of information from the root process to a destination process.
// The destination process is set as a variable in the code and must be less than the number of processes started.
//
// example usage:
//compile: mpicc -o mpi_isend mpi_isend.c
//run: mpirun -n 4 mpi_isend
//
int main(argc, argv)
     int argc;
     char **argv;
{
  int rank, size;
  int tag, destination, count;
  int buffer; //value to send

  tag = 1234;
  destination = 2; //destination process
  count = 1; //number of elements in buffer    

  MPI_Status status;
  MPI_Request request = MPI_REQUEST_NULL;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size); //number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); //rank of current process

  if (destination >= size) {
    MPI_Abort(MPI_COMM_WORLD, 1); // destination process must be under the number of processes created, otherwise abort
  }

  if (rank == 0) {
    printf("Enter a value to send to processor %d:\n", destination);
    scanf("%d", &buffer);
    MPI_Isend(&buffer, count, MPI_INT, destination, tag, MPI_COMM_WORLD, &request); //non blocking send to destination process
  }

  if (rank == destination) {
    MPI_Irecv(&buffer, count, MPI_INT, 0, tag, MPI_COMM_WORLD, &request); //destination process receives
  }

  MPI_Wait(&request, &status); //bloks and waits for destination process to receive data

  if (rank == 0) {
    printf("processor %d sent %d\n", rank, buffer);
  }
  if (rank == destination) {
    printf("processor %d got %d\n", rank, buffer);
  }

  MPI_Finalize();

  return 0;
}
