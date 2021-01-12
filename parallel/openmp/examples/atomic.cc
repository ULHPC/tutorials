#include <iostream>
#include <omp.h>

using namespace std;

int main()
{

  omp_set_num_threads(5);

  int total = 0;
  
#pragma omp parallel 
  {
    for(int i = 0; i < 10; i++)
      {
        // Atomically add one to the total
#pragma omp atomic // what happens if you remove the atomic 
        total++;      
      }
  }    
  cout <<"Total = " << total <<endl;

  return 0;
}
