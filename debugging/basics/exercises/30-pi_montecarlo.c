#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>

// Notes:
//
//  Add "-pthread -lm" on the compilation command line to link with the pthread and the math libraries
//
// About Monte Carlo Estimation for Pi:
// http://polymer.bu.edu/java/java/montepi/MontePi.html

int number_of_iterations = 1000000;
int number_of_threads = 4;


// return a random number in [0,1]
float get_unit_random_number()
{
   int random_value = rand(); //Generate a random number
   float unit_random = random_value / (float) RAND_MAX; //make it between 0 and 1
   return unit_random;
}

typedef struct
{
  int nb_iter;
  int in_circle_count;
} Result;

void *thread_start()
{
  // Result
  Result result;
  result.nb_iter = number_of_iterations;
  result.in_circle_count = 0;

  int iter;
  for ( iter = 0 ; iter < number_of_iterations ; iter++ )
  {
    // Generate a random point in a square
    float x = get_unit_random_number();
    float y = get_unit_random_number();

    // Distance from the origin
    float dist = sqrt((x*x) + (y*y));

    // Count if the point is in the unit circle
    if( dist < 1.0 )
    {
      result.in_circle_count++;
    }
  }

  pthread_exit((void*) &result);
}


int main(int argc, char** argv)
{
  // Get the number of iterations from the parameters
  if ( argc >= 2 )
  {
    number_of_iterations = atoi(argv[1]);
  }
  // Get the number of threads from the parameters
  if ( argc >= 3 )
  {
    number_of_threads = atoi(argv[2]);
  }



  // Create all threads
  pthread_t* threads = malloc( number_of_threads * sizeof(pthread_t) );
  int i;
  for( i = 0 ; i < number_of_threads ; i++ )
  {
    int res = pthread_create( &threads[i], NULL, thread_start, NULL );
    if (res)
    {
      printf("Error: pthread_create() failed with error code %d\n", res);
      exit(-1);
    }
  }

  // Wait for all the threads to finish and accumulate results
  Result all_results;
  for( i = 0 ; i < number_of_threads ; i++ )
  {
    void* status;
    int res = pthread_join(threads[i], &status);
    if (res)
    {
      printf("Error: pthread_join() failed with error code %d\n", res);
      exit(-1);
    }

    // Accumulate result
    Result* thread_result = (Result*) status;
    all_results.nb_iter += thread_result->nb_iter;
    all_results.in_circle_count += thread_result->in_circle_count;
  }

  double pi = 4 * all_results.in_circle_count / all_results.nb_iter;
  printf("Value for Pi after %d iteration is %f\n", all_results.nb_iter, pi);

  return 0;
}
