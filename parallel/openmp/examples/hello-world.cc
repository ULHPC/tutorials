#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  // printing out from master thread
  cout << "Hello world from master thread"<<endl;
  // creating the parallel region (team of threads)
#pragma omp parallel num_threads(5)
  {
    // optional for make serial run
    // #pragma omp critical
    cout << "Hello world from thread id "
	 << omp_get_thread_num() << " from the team size of "
	 << omp_get_num_threads()
	 << endl;
  } // parallel region is closed 
 cout << "end of the programme from master thread" << endl;
 return 0;
}
