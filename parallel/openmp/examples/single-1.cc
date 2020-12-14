#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  //omp_set_num_threads(3);
  // or
#pragma omp parallel num_threads(3) 
  {
#pragma omp single
    {
      cout <<"single 1 from thread id is " << omp_get_thread_num () 
	   <<endl;
    }
#pragma omp single
    {
      cout <<"single 2 from thread id is " << omp_get_thread_num () 
	   <<endl;
    }
#pragma omp single
    {
      cout <<"single 3 from thread id is "<< omp_get_thread_num () 
	   <<endl;
    }

  }
  return 0;
}

