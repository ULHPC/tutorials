#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  int th_id = 10;
  int nthreads;

#pragma omp parallel private(th_id) //num_threads(3)

  //th_id is declared above.  It is is specified as private; so each
  //thread will have its own copy of th_id
  {
    th_id = omp_get_thread_num();
    cout << "Hello World from thread " <<  th_id <<endl;
  }
  return 0;
}
