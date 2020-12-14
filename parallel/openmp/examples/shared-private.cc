#include <string>
#include <iostream>

using namespace std;

int main()
{
  string i = "x", j = "y";
  int k = 3;

  //#pragma omp parallel shared(b,c) num_threads(2) //shared(b) num_threads(2)
  //#pragma omp parallel private(i,k) shared(j) num_threads(4)
  //#pragma omp parallel private(i,k) num_threads(4)
  cout << "initial i value is="<< i << " j value is=" << j << " and k value is=" << k <<endl;
#pragma omp parallel default(shared) private(i,k) num_threads(2)
    {
      i += "a";
      j += "b";
      k += 7;
      cout << "i becomes " << i << "j becomes "
           << j << " and k becomes is "
           << k << endl;
    }
    cout << "unchanged k value is=" << k
         << " unchanged i value is=" << i
         << " changed j is=" << j << endl;
    return 0;
}



