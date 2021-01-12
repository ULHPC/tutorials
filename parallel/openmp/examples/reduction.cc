#include<iostream>
#include<omp.h>
#include<time.h>

using namespace std;

int main()
{
  int N=100000;
  int sum=0;
  double start = omp_get_wtime(); 
#pragma omp parallel for reduction(+:sum) num_threads(10)
  for (int i = 0; i < N; i++)
    sum += i;
  
  double end = omp_get_wtime(); 
  cout<< "Work took " << end - start << " seconds " <<endl; 
  
  cout << "total sum is " << sum << endl;
  return 0;
}



  
