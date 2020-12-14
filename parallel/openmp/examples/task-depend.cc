#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  int a=0, b=0, c=0;
  
#pragma omp parallel num_threads(10)
  {
#pragma omp single
    {
#pragma omp task depend(out:a)
      {
	a = 2;
	cout << "task 1" <<endl;
      }
#pragma omp task depend(out:c)
      {
        c = 4;
        cout <<"task 2" <<endl;
      }
#pragma omp task depend(in:a) depend(in:b) 
      {
        b = 3;
        a = 1;
        cout <<"task 3" <<endl;
      }
    }
  }
  cout << a << b << c <<endl;
}
