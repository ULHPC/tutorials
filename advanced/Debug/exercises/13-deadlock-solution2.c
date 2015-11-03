#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

// Notes:
//
//  Add -pthread on the compilation command line to link with the pthread library

#define NB_THREADS 2
#define NB_TRANSFER 1000000

typedef struct
{
  int balance;
} Account;


// Global lock
pthread_mutex_t mutex;


void deposit(Account* account, int amount)
{
  pthread_mutex_lock(&mutex);
  account->balance += amount;
  pthread_mutex_unlock(&mutex);
}


void transfer(Account* accountA, Account* accountB, int amount)
{
  pthread_mutex_lock(&mutex);
  accountA->balance += amount;
  accountB->balance -= amount;
  pthread_mutex_unlock(&mutex);
}


// Use two global account objects
Account accountA;
Account accountB;




void *thread_start()
{
  // Each thread will call deposit NB_DEPOSIT times
  int i;
  for ( i = 0 ; i < NB_TRANSFER ; i++)
  {
    // Transfer pesos between account A and B
    transfer( &accountA, &accountB, 10 );
    transfer( &accountB, &accountA, 10 );
  }

  pthread_exit(NULL);
}


int main(int argc, char** argv)
{
  // Set initial balance to 0
  accountA.balance = 1000;
  accountB.balance = 1000;
  pthread_mutex_init( &mutex, NULL );


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


  printf("Final balance on account A is %d\n", accountA.balance );
  printf("Final balance on account B is %d\n", accountB.balance );

  pthread_mutex_destroy(&mutex);
  pthread_exit(NULL);

  return 0;
}
