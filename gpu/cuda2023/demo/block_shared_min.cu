// Copyright 2023 Pierre Talbot

#include "../utility.hpp"
#include <algorithm>
#include <climit>

__global__ void parallel_min(int* v, size_t n, int* res) {
  __shared__ int* local_min;
  if(threadIdx.x == 0) {
    local_min = new int[blockDim.x];
    for(int i = 0; i < blockDim.x; ++i) {
      local_min[i] = INT_MAX;
    }
  }
  __syncthreads();
  for(size_t i = threadIdx.x; i < n; i += blockDim.x) {
    local_min[threadIdx.x] = min(local_min[threadIdx.x], v[i]);
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    *res = local_min[0];
    for(size_t i = 1; i < blockDim.x; ++i) {
      *res = min(*res, local_min[i]);
    }
  }
}

int main() {
  size_t n = 100000000;
  int* v = init_random_vector(n);
  int* res;
  CUDIE(cudaMallocManaged(&res, sizeof(int)));
  parallel_min<<<1, 256>>>(v, n, res);
  CUDIE(cudaDeviceSynchronize());
  std::cout << "Minimum: " << *res << std::endl;
  cudaFree(v);
  cudaFree(res);
}
