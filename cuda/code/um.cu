#include<iostream>
#include <string>
#include "cuda.h"
	
__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride) {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i) {
    a[i] = 1;
  }
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    std::cerr<<"Missing option [g|c|gc|cg]\n";
    exit(1);
  }
  std::string opt {argv[1]};
  std::cout<<"Option: "<<opt<<'\n';

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  /* 
   * Conduct experiments to learn more about the behavior of
   * 'cudaMallocManaged'.
   * 
   * What happens when unified memory is accessed only by the GPU?
   * What happens when unified memory is accessed only by the CPU? 
   * What happens when unified memory is accessed first by the GPU then the CPU?
   * What happens when unified memory is accessed first by the CPU then the GPU?
   * 
   * Hypothesize about UM behavior, page faulting specificially, before each
   * experiement, and then verify by running 'nvprof'.
   */
  if (opt == "g") {
    deviceKernel<<<80,512>>>(a, N);
    cudaDeviceSynchronize();
  } else if (opt == "c") {
    hostFunction(a, N);
  } else if (opt == "gc") {
    deviceKernel<<<80,512>>>(a, N);
    cudaDeviceSynchronize();
    hostFunction(a, N);
  } else if (opt == "cg") {
    hostFunction(a, N);
    deviceKernel<<<80,512>>>(a, N);
    cudaDeviceSynchronize();
  } else {
    std::cerr<<"Invalid option [g|c|gc|cg]: "<<opt<<'\n';
  }
  cudaFree(a);
}

