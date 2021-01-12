#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  int n=3;
  // for private varibale test
  int var = 5;
cout << "initial var value " << var <<endl;
#pragma omp parallel for lastprivate(var) num_threads(n) 
  for(int i = 0; i < n; i++)
    {
      var = i;
      // var += omp_get_thread_num();
      cout << " lastprivate in parallel region " << var << endl;
    }
cout << "lastprivate after parallel region " << var <<endl;
  return 0;
}
