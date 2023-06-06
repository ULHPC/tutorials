#include "../utility.hpp"
#include <climits>

__global__ void parallel_min(int* v, size_t n, int* local_min) {
  local_min[threadIdx.x] = INT_MAX;
  size_t m = n / blockDim.x + (n % blockDim.x != 0);
  size_t from = threadIdx.x * m;
  size_t to = min(n, from + m);
  for(size_t i = from; i < to; ++i) {
    local_min[threadIdx.x] = min(local_min[threadIdx.x], v[i]);
  }
}

__global__ void parallel_min_stride(int* v, size_t n, int* local_min) {
  local_min[threadIdx.x] = INT_MAX;
  for(size_t i = threadIdx.x; i < n; i += blockDim.x) {
    local_min[threadIdx.x] = min(local_min[threadIdx.x], v[i]);
  }
}

int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "usage: " << argv[0] << " <vector size> <threads-per-block>" << std::endl;
    std::cout << "example: " << argv[0] << " 1000000000 512" << std::endl;
    exit(1);
  }
  size_t n = std::stoi(argv[1]);
  size_t threads_per_block = std::stoi(argv[2]);

  // I. Initialize and allocate in managed memory the array of numbers.
  int* v = init_random_vector(n);
  int* local_min;
  CUDIE(cudaMallocManaged(&local_min, sizeof(int) * threads_per_block));

  // II. Run the kernel on one block, every thread `i` stores its local minimum in `local_min[i]`.
  long gpu_ms = benchmark_ms([&]{
    parallel_min<<<1, threads_per_block>>>(v, n, local_min);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU: " << gpu_ms << " ms" << std::endl;

  long gpu_strided_ms = benchmark_ms([&]{
    parallel_min_stride<<<1, threads_per_block>>>(v, n, local_min);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU (contiguous memory accesses): " << gpu_strided_ms << " ms" << std::endl;

  // III. Find the true minimum among all local minimum computed on the GPU.
  int res = local_min[0];
  for(size_t i = 1; i < threads_per_block; ++i) {
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
