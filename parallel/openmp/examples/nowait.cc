#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  cout <<" master thread" << endl;
#pragma omp parallel num_threads(5)
  {
    // what happens if you remove the nowait. 
#pragma omp for nowait
    for (int i = 0; i < 10; ++i) 
      {
      cout << " no wait" <<endl;
      }
    cout << " parallel" <<endl;
  }
}
