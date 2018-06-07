#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

// Notes:
//
//  Add -pthread on the compilation command line to link with the pthread library


#define NB_THREADS 8
#define NB_DEPOSIT 1000000

typedef struct
{
  int balance;
} Account;


void deposit(Account* account, int amount)
{
  account->balance += amount;
}




// Use one global account object
Account account;






void *thread_start()
{
  // Each thread will call deposit NB_DEPOSIT times
  int i;
  for ( i = 0 ; i < NB_DEPOSIT ; i++)
  {
    // Add one pesos to the account
    deposit( &account, 1 );
  }

  pthread_exit(NULL);
}


int main(int argc, char** argv)
{
  // Set initial balance to 0
  account.balance = 0;


  pthread_t threads[NB_THREADS];

  // Create all thread
  int i;
  for( i = 0 ; i < NB_THREADS ; i++ )
  {
    int res = pthread_create( &threads[i], NULL, thread_start, NULL );
    if (res)
    {
      printf("Error: pthread_create() failed with error code %d\n", res);
      exit(-1);
    }
  }

  // Wait for all the threads to finish
  for( i = 0 ; i < NB_THREADS ; i++ )
  {
    int res = pthread_join(threads[i], NULL);
    if (res)
    {
      printf("Error: pthread_join() failed with error code %d\n", res);
      exit(-1);
    }
  }


  printf("Final balance is %d\n", account.balance );

  pthread_exit(NULL);

  return 0;
}
