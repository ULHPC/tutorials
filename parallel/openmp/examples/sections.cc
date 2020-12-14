#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
#pragma omp parallel num_threads(6) 
  {
#pragma omp sections
    {
      cout <<"sections from thread id " << omp_get_thread_num () 
	   <<endl;
#pragma omp section
      {
        cout <<"section from thread id " << omp_get_thread_num () 
	     <<endl;
      }
#pragma omp section
      {
        cout <<"section from thread id " << omp_get_thread_num () 
	     << endl;
      }
#pragma omp section
      {
        cout <<"section from thread id " << omp_get_thread_num ()
	     <<endl;
      }
    }
  }
  return 0;
}
