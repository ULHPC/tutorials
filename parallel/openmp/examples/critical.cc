#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  int global_sum=0;
#pragma omp parallel num_threads(10)
  {
#pragma omp for 
    for(int i=0; i < 10; i++)
      {
        int local_sum=i;

	// what happens if you remove the critical 
#pragma omp critical 

        if(global_sum <local_sum)
          {
            global_sum=local_sum;
            cout << " I am in critical" <<endl;
          }
      }
  }
  cout << "global sum is "<< global_sum << endl;
  return 0;
}
