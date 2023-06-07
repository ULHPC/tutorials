// Copyright 2023 Pierre Talbot

#include "../utility.hpp"
#include <climits>

__device__ void block_min(int* v, size_t begin, size_t end, int* local_min) {
  size_t id = threadIdx.x + (blockDim.x * blockIdx.x);
  local_min[id] = INT_MAX;
  size_t n = end - begin;
  size_t m = n / blockDim.x + (n % blockDim.x != 0);
  size_t from = begin + threadIdx.x * m;
  size_t to = min(end, from + m);
  for(size_t i = from; i < to; ++i) {
    local_min[id] = min(local_min[id], v[i]);
  }
}

__device__ void block_min_stride(int* v, size_t begin, size_t end, int* local_min) {
  size_t id = threadIdx.x + (blockDim.x * blockIdx.x);
  local_min[id] = INT_MAX;
  for(size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    local_min[id] = min(local_min[id], v[i]);
  }
}

template <bool stride>
__global__ void grid_min(int* v, size_t n, int* local_min) {
  size_t m = n / gridDim.x + (n % gridDim.x != 0);
  size_t begin = blockIdx.x * m;
  size_t end = min(n, begin + m);
  if constexpr(stride) {
    block_min_stride(v, begin, end, local_min);
  }
  else {
    block_min(v, begin, end, local_min);
  }
}

int main(int argc, char** argv) {
  if(argc != 4) {
    std::cout << "usage: " << argv[0] << " <vector size> <threads-per-block> <num-blocks>" << std::endl;
    std::cout << "example: " << argv[0] << " 1000000000 256 256" << std::endl;
    exit(1);
  }
  size_t n = std::stoi(argv[1]);
  size_t threads_per_block = std::stoi(argv[2]);
  size_t num_blocks = std::stoi(argv[3]);

  // I. Initialize and allocate in managed memory the array of numbers.
  int* v = init_random_vector(n);
  int* local_min;
  CUDIE(cudaMallocManaged(&local_min, sizeof(int) * num_blocks * threads_per_block));

  // II. Run the kernel on one block, every thread `i` stores its local minimum in `local_min[i]`.
  long gpu_ms = benchmark_ms([&]{
    grid_min<false><<<num_blocks, threads_per_block>>>(v, n, local_min);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU: " << gpu_ms << " ms" << std::endl;

  long gpu_strided_ms = benchmark_ms([&]{
    grid_min<true><<<num_blocks, threads_per_block>>>(v, n, local_min);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU (contiguous memory accesses): " << gpu_strided_ms << " ms" << std::endl;

  // III. Find the true minimum among all local minimum computed on the GPU.
  int res = local_min[0];
  for(size_t i = 1; i < num_blocks * threads_per_block; ++i) {
    res = min(res, local_min[i]);
  }
  std::cout << "Minimum on GPU: " << res << std::endl;

  // IV. Find the minimum using CPU.
  int cpu_res = INT_MAX;
  for(size_t i = 0; i < n; ++i) {
    cpu_res = std::min(cpu_res, v[i]);
  }
  std::cout << "Minimum on CPU: " << cpu_res << std::endl;

  cudaFree(v);
  cudaFree(local_min);
}
