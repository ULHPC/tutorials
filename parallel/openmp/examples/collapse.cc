#include<iostream>
#include<omp.h>
#include<time.h>

using namespace std;

int main()
{
  int N=1000;
  // initialize the array
  int *a = new int [N];  
  int *b = new int [N];  
  int *c = new int [N];
  
  for (int i = 0; i < N; i++)
    {
      a[i]=i;
      b[i]=i;
    }
  // number of threads can be also set in here
  // omp_set_num_threads(5);
  
  double start = omp_get_wtime(); 

  // comment out this line and see the performance 
#pragma omp parallel for collapse(2) num_threads(5)
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
    {
      c[i]=a[i]*b[j];
    }

  double end = omp_get_wtime(); 
  cout<<" Work took " << end - start << " seconds " <<endl; 

  /*
  for (int i = 0; i < N; i++)
    {
      cout << c[i] << endl;
    }
  */

  delete[] a;
  delete[] b;
  delete[] c;

  return 0;
}


  
