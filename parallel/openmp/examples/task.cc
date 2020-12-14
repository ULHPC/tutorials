#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[])
{
#pragma omp parallel num_threads(10)
  {
#pragma omp single
    {
#pragma omp task
        cout << "Task-1 executed by thread " << omp_get_thread_num() <<endl;
#pragma omp task
        cout << "Task-2 executed by thread " << omp_get_thread_num() <<endl;
//what happens if you remove the taskwait 
#pragma omp taskwait
#pragma omp task
        cout << "Task-3 executed by thread " << omp_get_thread_num() <<endl;
    }
    //    cout << "final" <<endl;
  }
}
