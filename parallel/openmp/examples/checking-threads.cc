#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  // set number of the threads
  omp_set_num_threads(10);
  // or you can directly set it in here as well
  // #pragma omp parallel //num_threads(5)
  cout << "maximum threads set earlier "<< omp_get_max_threads()<< endl;
#pragma omp parallel 
  {
    cout << "get a thread id "
	 << omp_get_thread_num() << " from the team size of "
	 << omp_get_num_threads() << " number of process " << omp_get_num_procs()
	 << endl;
  } // parallel region is closed 

 return 0;
}
