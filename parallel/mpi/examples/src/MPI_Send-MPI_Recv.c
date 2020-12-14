#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main ( int argc , char *argv [])   {
  int numtasks , rank , dest , source , rc , count , len ;  
  char inmsg[30],inmsg1 [30], outmsg0 []= "Hello Task 1" ,
    outmsg1 []= "You are Welcome Task 0" , outmsg2 [ ]= "Another message from Task 0" ;
  int nummsg1 = 10 , nummsg2 = 7 , nummsg3 ;   /*Just Random int s.*/
  int Tag_char = 27 , Tag_int = 15 ;   /*Just Random int s. There is a max size consideration.*/
  char name [MPI_MAX_PROCESSOR_NAME];
  MPI_Init (&argc ,&argv );
  MPI_Comm_size ( MPI_COMM_WORLD , &numtasks );
  MPI_Comm_rank ( MPI_COMM_WORLD , &rank );
  MPI_Get_processor_name ( name , &len );
  if (rank==0) {      
    dest = 1 ;
    source = 1 ;
    MPI_Send (&outmsg0 , strlen ( outmsg0 ), MPI_CHAR , dest , Tag_char ,MPI_COMM_WORLD );
    MPI_Send (&nummsg1 , 1 , MPI_INT , dest , Tag_int , MPI_COMM_WORLD );
    MPI_Send (&outmsg2 , strlen ( outmsg2 ), MPI_CHAR , dest , Tag_char ,MPI_COMM_WORLD );
    printf( "\nTask 0 on processor %s has sent its  messages to Task 1. \n" , name );
    MPI_Recv (&nummsg3 , 1 , MPI_INT , source , Tag_int , MPI_COMM_WORLD,MPI_STATUS_IGNORE );
    memset(inmsg, 0, 30);
    MPI_Recv(&inmsg , 30 , MPI_CHAR , source , Tag_char , MPI_COMM_WORLD ,MPI_STATUS_IGNORE );
    printf( "Task 0 received this  message from Task 1:   %s\n" , inmsg );
    printf( "Task 0 also received this  message:   %i\n\n" , nummsg3 );
  } else if( rank == 1 ) {
    dest = 0 ;
    source = 0 ;
    memset(inmsg, 0, 30);
    MPI_Recv (&inmsg , 30 , MPI_CHAR , source , Tag_char , MPI_COMM_WORLD ,   MPI_STATUS_IGNORE );
    MPI_Recv (&nummsg3 , 1 , MPI_INT , source , Tag_int , MPI_COMM_WORLD ,   MPI_STATUS_IGNORE );
    printf ( "Task 1 on processor %s received this  message:   %s\n" , name , inmsg );
    memset(inmsg1, 0, 30);
    MPI_Recv (&inmsg1 , 30 , MPI_CHAR , source , Tag_char , MPI_COMM_WORLD ,   MPI_STATUS_IGNORE );
    printf ( "Task 1 also received this  message:   %i\n" , nummsg3 );
    printf ( "Task 1 received this  message as well: %s\n" , inmsg1 );
    printf ( "Task 1 has not sent its  message to Task 0 yet.\n\n" );
    MPI_Send (&outmsg1 , strlen (outmsg1), MPI_CHAR , dest , Tag_char , MPI_COMM_WORLD );
    MPI_Send (&nummsg2 , 1 , MPI_INT , dest , Tag_int , MPI_COMM_WORLD );
  }
  MPI_Finalize ();
}
