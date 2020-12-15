#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  int c=2;
  int N=10000;
  int *a=new int[N];
  int *b=new int[N];

  for(int i=0; i < N; i++)
    {
      a[i] = i;
      b[i] = i;
    }
  omp_set_num_threads(10);

  double start = omp_get_wtime(); 
#pragma omp parallel for //simd 
    for(int i = 0; i < N; i++)
      {
	a[i] = a[i] + b[i];
      }
  double end = omp_get_wtime(); 
  cout<<" Work took " << end - start << " seconds " <<endl; 
  /*
  for(int i=0; i < N; i++)
    {
      cout << a[i] << endl;
      //cout << b[i] << endl;
    }
  */
  return 0;
}
