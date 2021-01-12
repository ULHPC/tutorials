# include <math.h>
# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>

double L = 1.0;			/* linear size of square region */
int N = 32;			/* number of interior points per dim */

double *u, *u_new;		/* linear arrays to hold solution */

/* macro to index into a 2-D (N+2)x(N+2) array */
#define INDEX(i,j) ((N+2)*(i)+(j))

int my_rank;			/* rank of this process */

int *proc;			/* process indexed by vertex */
int *i_min, *i_max;		/* min, max vertex indices of processes */
int *left_proc, *right_proc;	/* processes to left and right */

/*
  Functions:
*/
int main ( int argc, char *argv[] );
void allocate_arrays ( );
void jacobi ( int num_procs, double f[] );
void make_domains ( int num_procs );
double *make_source ( );
void timestamp ( );

/******************************************************************************/

int main ( int argc, char *argv[] ) 

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for POISSON_MPI.

  Discussion:

    This program solves Poisson's equation in a 2D region.

    The Jacobi iterative method is used to solve the linear system.

    MPI is used for parallel execution, with the domain divided
    into strips.

  Modified:

    22 September 2013

  Local parameters:

    Local, double F[(N+2)x(N+2)], the source term.

    Local, int N, the number of interior vertices in one dimension.

    Local, int NUM_PROCS, the number of MPI processes.

    Local, double U[(N+2)*(N+2)], a solution estimate.

    Local, double U_NEW[(N+2)*(N+2)], a solution estimate.
*/
{
  double change;
  double epsilon = 1.0E-03;
  double *f;
  char file_name[100];
  int i;
  int j;
  double my_change;
  int my_n;
  int n;
  int num_procs;
  int step;
  double *swap;
  double wall_time;
/*
  MPI initialization.
*/
  MPI_Init ( &argc, &argv );

  MPI_Comm_size ( MPI_COMM_WORLD, &num_procs );

  MPI_Comm_rank ( MPI_COMM_WORLD, &my_rank );
/*
  Read commandline arguments, if present.
*/
  if ( 1 < argc )
  {
    sscanf ( argv[1], "%d", &N );
  }
  else
  {
    N = 32;
  }

  if ( 2 < argc )
  {
    sscanf ( argv[2], "%lf", &epsilon );
  }
  else
  {
    epsilon = 1.0E-03;
  }
  if ( 3 < argc )
  {
    strcpy ( file_name, argv[3] );
  }
  else
  {
    strcpy ( file_name, "poisson_mpi.out" );
  }
/*
  Print out initial information.
*/
  if ( my_rank == 0 ) 
  {
    timestamp ( );
    printf ( "\n" );
    printf ( "POISSON_MPI:\n" );
    printf ( "  C version\n" );
    printf ( "  2-D Poisson equation using Jacobi algorithm\n" );
    printf ( "  ===========================================\n" );
    printf ( "  MPI version: 1-D domains, non-blocking send/receive\n" );
    printf ( "  Number of processes         = %d\n", num_procs );
    printf ( "  Number of interior vertices = %d\n", N );
    printf ( "  Desired fractional accuracy = %f\n", epsilon );
    printf ( "\n" );
  }

  allocate_arrays ( );
  f = make_source ( );
  make_domains ( num_procs );

  step = 0;
/*
  Begin timing.
*/
  wall_time = MPI_Wtime ( );
/*
  Begin iteration.
*/
  do 
  {
    jacobi ( num_procs, f );
    ++step;
/* 
  Estimate the error 
*/
    change = 0.0;
    n = 0;

    my_change = 0.0;
    my_n = 0;

    for ( i = i_min[my_rank]; i <= i_max[my_rank]; i++ )
    {
      for ( j = 1; j <= N; j++ )
      {
        if ( u_new[INDEX(i,j)] != 0.0 ) 
        {
          my_change = my_change 
            + fabs ( 1.0 - u[INDEX(i,j)] / u_new[INDEX(i,j)] );

          my_n = my_n + 1;
        }
      }
    }
    MPI_Allreduce ( &my_change, &change, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD );

    MPI_Allreduce ( &my_n, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    if ( n != 0 )
    {
      change = change / n;
    }
    if ( my_rank == 0 && ( step % 10 ) == 0 ) 
    {
      printf ( "  N = %d, n = %d, my_n = %d, Step %4d  Error = %g\n", 
        N, n, my_n, step, change );
    }
/* 
  Interchange U and U_NEW.
*/
    swap = u;
    u = u_new;
    u_new = swap;
  } while ( epsilon < change );

/* 
  Here is where you can copy the solution to process 0 
  and print to a file.
*/

/*
  Report on wallclock time.
*/
  wall_time = MPI_Wtime() - wall_time;
  if ( my_rank == 0 )
  {
    printf ( "\n" );
    printf ( "  Wall clock time = %f secs\n", wall_time );
  }
/*
  Terminate MPI.
*/
  MPI_Finalize ( );
/*
  Free memory.
*/
  free ( f );
/*
  Terminate.
*/
  if ( my_rank == 0 )
  {
    printf ( "\n" );
    printf ( "POISSON_MPI:\n" );
    printf ( "  Normal end of execution.\n" );
    printf ( "\n" );
    timestamp ( );
  }
 
  return 0;
}
/******************************************************************************/

void allocate_arrays ( ) 

/******************************************************************************/
/*
  Purpose:

    ALLOCATE_ARRAYS creates and zeros out the arrays U and U_NEW.

  Modified:

    10 September 2013
*/
{
  int i;
  int ndof;

  ndof = ( N + 2 ) * ( N + 2 );

  u = ( double * ) malloc ( ndof * sizeof ( double ) );
  for ( i = 0; i < ndof; i++)
  {
    u[i] = 0.0;
  }

  u_new = ( double * ) malloc ( ndof * sizeof ( double ) );
  for ( i = 0; i < ndof; i++ )
  {
    u_new[i] = 0.0;
  }

  return;
}
/******************************************************************************/

void jacobi ( int num_procs, double f[] ) 

/******************************************************************************/
/*
  Purpose:

    JACOBI carries out the Jacobi iteration for the linear system.

  Modified:

    16 September 2013

  Parameters:

    Input, int NUM_PROCS, the number of processes.

    Input, double F[(N+2)*(N+2)], the right hand side of the linear system.
*/
{
  double h;
  int i;
  int j;
  MPI_Request request[4];
  int requests;
  MPI_Status status[4];
/*
  H is the lattice spacing.
*/
  h = L / ( double ) ( N + 1 );
/* 
  Update ghost layers using non-blocking send/receive 
*/
  requests = 0;

  if ( left_proc[my_rank] >= 0 && left_proc[my_rank] < num_procs ) 
  {
    MPI_Irecv ( u + INDEX(i_min[my_rank] - 1, 1), N, MPI_DOUBLE,
      left_proc[my_rank], 0, MPI_COMM_WORLD,
      request + requests++ );

    MPI_Isend ( u + INDEX(i_min[my_rank], 1), N, MPI_DOUBLE,
      left_proc[my_rank], 1, MPI_COMM_WORLD,
      request + requests++ );
  }

  if ( right_proc[my_rank] >= 0 && right_proc[my_rank] < num_procs ) 
  {
    MPI_Irecv ( u + INDEX(i_max[my_rank] + 1, 1), N, MPI_DOUBLE,
      right_proc[my_rank], 1, MPI_COMM_WORLD,
      request + requests++ );

    MPI_Isend ( u + INDEX(i_max[my_rank], 1), N, MPI_DOUBLE,
      right_proc[my_rank], 0, MPI_COMM_WORLD,
      request + requests++ );
  }
/* 
  Jacobi update for internal vertices in my domain.
*/
  for ( i = i_min[my_rank] + 1; i <= i_max[my_rank] - 1; i++ )
  {
    for ( j = 1; j <= N; j++ )
    {
      u_new[INDEX(i,j)] =
        0.25 * ( u[INDEX(i-1,j)] + u[INDEX(i+1,j)] +
                 u[INDEX(i,j-1)] + u[INDEX(i,j+1)] +
                 h * h * f[INDEX(i,j)] );
    }
  }
/* 
  Wait for all non-blocking communications to complete.
*/
  MPI_Waitall ( requests, request, status );
/* 
  Jacobi update for boundary vertices in my domain.
*/
  i = i_min[my_rank];
  for ( j = 1; j <= N; j++ )
  {
    u_new[INDEX(i,j)] =
      0.25 * ( u[INDEX(i-1,j)] + u[INDEX(i+1,j)] +
               u[INDEX(i,j-1)] + u[INDEX(i,j+1)] +
               h * h * f[INDEX(i,j)] );
  }

  i = i_max[my_rank];
  if (i != i_min[my_rank])
  {
    for (j = 1; j <= N; j++)
    {
      u_new[INDEX(i,j)] =
        0.25 * ( u[INDEX(i-1,j)] + u[INDEX(i+1,j)] +
                 u[INDEX(i,j-1)] + u[INDEX(i,j+1)] +
                 h * h * f[INDEX(i,j)] );
    }
  }

  return;
}
/******************************************************************************/

void make_domains ( int num_procs ) 

/******************************************************************************/
/*
  Purpose:

    MAKE_DOMAINS sets up the information defining the process domains.

  Modified:

    10 September 2013

  Parameters:

    Input, int NUM_PROCS, the number of processes.
*/
{
  double d;
  double eps;
  int i;
  int p;
  double x_max;
  double x_min;
/* 
  Allocate arrays for process information.
*/
  proc = ( int * ) malloc ( ( N + 2 ) * sizeof ( int ) );
  i_min = ( int * ) malloc ( num_procs * sizeof ( int ) );
  i_max = ( int * ) malloc ( num_procs * sizeof ( int ) );
  left_proc = ( int * ) malloc ( num_procs * sizeof ( int ) );
  right_proc = ( int * ) malloc ( num_procs * sizeof ( int ) );
/* 
  Divide the range [(1-eps)..(N+eps)] evenly among the processes.
*/
  eps = 0.0001;
  d = ( N - 1.0 + 2.0 * eps ) / ( double ) num_procs;

  for ( p = 0; p < num_procs; p++ )
  {
/* 
  The I indices assigned to domain P will satisfy X_MIN <= I <= X_MAX.
*/
    x_min = - eps + 1.0 + ( double ) ( p * d );
    x_max = x_min + d;
/* 
  For the node with index I, store in PROC[I] the process P it belongs to.
*/
    for ( i = 1; i <= N; i++ )
    {
      if ( x_min <= i && i < x_max )
      {
        proc[i] = p;
      }
    }
  }
/* 
  Now find the lowest index I associated with each process P.
*/
  for ( p = 0; p < num_procs; p++ )
  {
    for ( i = 1; i <= N; i++ )
    {
      if ( proc[i] == p )
      {
        break;
      }
    }
    i_min[p] = i;
/* 
  Find the largest index associated with each process P.
*/
    for ( i = N; 1 <= i; i-- )
    {
      if ( proc[i] == p )
      {
        break;
      }
    }
    i_max[p] = i;
/* 
  Find the processes to left and right. 
*/
    left_proc[p] = -1;
    right_proc[p] = -1;

    if ( proc[p] != -1 ) 
    {
      if ( 1 < i_min[p] && i_min[p] <= N )
      {
        left_proc[p] = proc[i_min[p] - 1];
      }
      if ( 0 < i_max[p] && i_max[p] < N )
      {
        right_proc[p] = proc[i_max[p] + 1];
      }
    }
  }

  return;
}
/******************************************************************************/

double *make_source ( ) 

/******************************************************************************/
/*
  Purpose:

    MAKE_SOURCE sets up the source term for the Poisson equation.

  Modified:

    16 September 2013

  Parameters:

    Output, double *MAKE_SOURCE, a pointer to the (N+2)*(N+2) source term
    array.
*/
{
  double *f;
  int i;
  int j;
  int k;
  double q;

  f = ( double * ) malloc ( ( N + 2 ) * ( N + 2 ) * sizeof ( double ) );

  for ( i = 0; i < ( N + 2 ) * ( N + 2 ); i++ )
  {
    f[i] = 0.0;
  }
/* 
  Make a dipole.
*/
  q = 10.0;

  i = 1 + N / 4;
  j = i;
  k = INDEX ( i, j );
  f[k] = q;

  i = 1 + 3 * N / 4;
  j = i;
  k = INDEX ( i, j );
  f[k] = -q;

  return f;
}
/******************************************************************************/

void timestamp ( )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
