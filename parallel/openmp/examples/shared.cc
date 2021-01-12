#include<iostream>
#include<omp.h>

using namespace std;

int main()
{
  int x = 5;
#pragma omp parallel num_threads(3) private(x)
  {
    //int x = 5;
    //  x = x+1;
    cout << "shared: x is " << x << endl;
  }

  return 0;
}
