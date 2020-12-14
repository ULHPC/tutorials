/**
 * @author RookieHPC
 * @brief Original source code at https://www.rookiehpc.com/mpi/docs/mpi_ssend.php
 **/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
 
/**
 * @brief Illustrates how to send a message in a blocking synchronous fashion.
 * @details This program is meant to be run with 2 processes: a sender and a
 * receiver.
 **/
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
 
  // Get the number of processes and check only 2 processes are used
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size != 2)
    {
      printf("This application is meant to be run with 2 processes.\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
  // Get my rank and do the corresponding job
  enum role_ranks { SENDER, RECEIVER };
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  switch(my_rank)
    {
    case SENDER:
      {
	// The "master" MPI process issues the MPI_Ssend.
	int buffer_sent = 12345;
	MPI_Request request;
	printf("MPI process %d sends value %d.\n", my_rank, buffer_sent);
	MPI_Ssend(&buffer_sent, 1, MPI_INT, RECEIVER, 0, MPI_COMM_WORLD);
	break;
      }
    case RECEIVER:
      {
	// The "slave" MPI process receives the message.
	int received;
	MPI_Recv(&received, 1, MPI_INT, SENDER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("MPI process %d received value: %d.\n", my_rank, received);
	break;
      }
    }
 
  MPI_Finalize();
 
  return EXIT_SUCCESS;
}
