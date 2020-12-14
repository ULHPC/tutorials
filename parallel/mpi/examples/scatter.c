/**
 * @author RookieHPC
 * @brief Original source code at https://www.rookiehpc.com/mpi/docs/mpi_scatter.php
 **/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
 
/**
 * @brief Illustrates how to use a scatter.
 * @details This application is meant to be run with 4 processes. Process 0 is
 * designed as root and begins with a buffer containing all values, and prints
 * them. It then dispatches these values to all the processes in the same
 * communicator. Other processes just receive the dispatched value meant for 
 * them. Finally, everybody prints the value received.
 *
 *                +-----------------------+
 *                |       Process 0       |
 *                +-----+-----+-----+-----+
 *                |  0  | 100 | 200 | 300 |
 *                +-----+-----+-----+-----+
 *                 /      |       |      \
 *                /       |       |       \
 *               /        |       |        \
 *              /         |       |         \
 *             /          |       |          \
 *            /           |       |           \
 * +-----------+ +-----------+ +-----------+ +-----------+
 * | Process 0 | | Process 1 | | Process 2 | | Process 3 |
 * +-+-------+-+ +-+-------+-+ +-+-------+-+ +-+-------+-+ 
 *   | Value |     | Value |     | Value |     | Value |   
 *   |   0   |     |  100  |     |  200  |     |  300  |   
 *   +-------+     +-------+     +-------+     +-------+   
 *                
 **/
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
 
  // Get number of processes and check that 4 processes are used
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size != 4)
    {
      printf("This application is meant to be run with 4 processes.\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
  // Determine root's rank
  int root_rank = 0;
 
  // Get my rank
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 
  // Define my value
  int my_value;
 
  if(my_rank == root_rank)
    {
      int buffer[4] = {0, 100, 200, 300};
      printf("Values to scatter from process %d: %d, %d, %d, %d.\n", my_rank, buffer[0], buffer[1], buffer[2], buffer[3]);
      MPI_Scatter(buffer, 1, MPI_INT, &my_value, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    }
  else
    {
      MPI_Scatter(NULL, 1, MPI_INT, &my_value, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    }
 
  printf("Process %d received value = %d.\n", my_rank, my_value);
 
  MPI_Finalize();
 
  return EXIT_SUCCESS;
}
