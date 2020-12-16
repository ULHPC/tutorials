#include<iostream>
#include<omp.h>
#include<time.h>

using namespace std;


int my_task(int a)
{
  return a*a;
}

int main()
{
  omp_set_num_threads(10);
  double start = omp_get_wtime(); 
  int *x=new int[10];
#pragma omp parallel 
  {
    int tid = omp_get_thread_num();
    x[tid] = my_task(tid);
    cout <<" before barrier" << x[tid] << endl;
    // what happens if you remove barrier 
#pragma omp barrier
    cout <<" after barrier" << x[tid] << endl;
  }
  return 0;
}


  
