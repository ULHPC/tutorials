#include<iostream>
#include<omp.h>
#include<time.h>

using namespace std;

int main()
{
  int N=100;
  int sum=0;

  omp_set_num_threads(10);
  double start = omp_get_wtime(); 

#pragma omp parallel 
  {
#pragma omp for nowait reduction(+:sum)
      for (int i = 0; i < N; i++)
	sum += i;

#pragma omp single
      {
	cout << "first sum is " << sum << endl;
      }
#pragma omp for reduction(+:sum) nowait
      for (int i = 0; i < N; i++)
	sum += i;
#pragma omp single
      {
	cout << "second sum is " << sum << endl;
      }
  }

    double end = omp_get_wtime(); 
  cout<< "Work took " << end - start << " seconds " <<endl; 
  
  cout << "second sum is " << sum << endl;
  return 0;
}


  
