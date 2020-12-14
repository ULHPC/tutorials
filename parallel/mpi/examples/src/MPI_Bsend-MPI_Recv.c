/**
 * @author RookieHPC
 * @brief Original source code at https://www.rookiehpc.com/mpi/docs/mpi_bsend.php
 **/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
 
/**
 * @brief Illustrates how to send a message in a blocking asynchronous fashion.
 * @details This application is meant to be used with 2 processes; 1 sender and
 * 1 receiver. The sender will declare a buffer containing enough space for 1
 * message that will contain 1 integer. It then attaches the buffer to MPI and
 * issues the MPI_Bsend. Finally, it detaches the buffer and frees it, while the
 * receiver prints the message received.
 **/
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
 
  // Get the number of processes and check only 2 are used.
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
	// Declare the buffer and attach it
	int buffer_attached_size = MPI_BSEND_OVERHEAD + sizeof(int);
	char* buffer_attached = (char*)malloc(buffer_attached_size);
	MPI_Buffer_attach(buffer_attached, buffer_attached_size);
 
	// Issue the MPI_Bsend
	int buffer_sent = 12345;
	printf("[MPI process %d] I send value %d.\n", my_rank, buffer_sent);
	MPI_Bsend(&buffer_sent, 1, MPI_INT, RECEIVER, 0, MPI_COMM_WORLD);
 
	// Detach the buffer. It blocks until all messages stored are sent.
	MPI_Buffer_detach(&buffer_attached, &buffer_attached_size);
	free(buffer_attached);
	break;
      }
    case RECEIVER:
      {
	// Receive the message and print it.
	int received;
	MPI_Recv(&received, 1, MPI_INT, SENDER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("[MPI process %d] I received value: %d.\n", my_rank, received);
	break;
      }
    }
 
  MPI_Finalize();
 
  return EXIT_SUCCESS;
}
