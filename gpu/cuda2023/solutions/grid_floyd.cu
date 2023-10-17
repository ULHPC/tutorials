// Copyright 2023 Pierre Talbot

#include "../utility.hpp"
#include <string>

__global__ void floyd_warshall_gpu(int** d, size_t n, size_t k) {
  for(int i = blockIdx.x; i < n; i += gridDim.x) {
    for(int j = threadIdx.x; j < n; j += blockDim.x) {
      if(d[i][j] > d[i][k] + d[k][j]) {
        d[i][j] = d[i][k] + d[k][j];
      }
    }
  }
}

void floyd_warshall_cpu(std::vector<std::vector<int>>& d) {
  size_t n = d.size();
  for(int k = 0; k < n; ++k) {
    for(int i = 0; i < n; ++i) {
      for(int j = 0; j < n; ++j) {
        if(d[i][j] > d[i][k] + d[k][j]) {
          d[i][j] = d[i][k] + d[k][j];
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  if(argc != 4) {
    std::cout << "usage: " << argv[0] << " <matrix size> <threads-per-block> <num-blocks>" << std::endl;
    exit(1);
  }
  size_t n = std::stoi(argv[1]);
  size_t threads_per_block = std::stoi(argv[2]);
  size_t num_blocks = std::stoi(argv[3]);

  // I. Generate a random distance matrix of size N x N.
  std::vector<std::vector<int>> cpu_distances = initialize_distances(n);
  // Note that `std::vector` cannot be used on GPU, hence we transfer it into a simple `int**` array in managed memory.
  int** gpu_distances = initialize_gpu_distances(cpu_distances);

  // II. Running Floyd Warshall on CPU.
  long cpu_ms = benchmark_one_ms([&]{
    floyd_warshall_cpu(cpu_distances);
  });
  std::cout << "CPU: " << cpu_ms << " ms" << std::endl;

  // III. Running Floyd Warshall on the whole GPU grid.
  long gpu_ms = benchmark_one_ms([&]{
    for(int k = 0; k < n; ++k) {
      floyd_warshall_gpu<<<num_blocks, threads_per_block>>>(gpu_distances, n, k);
      CUDIE(cudaDeviceSynchronize());
    }
  });
  std::cout << "GPU: " << gpu_ms << " ms" << std::endl;

  // IV. Verifying both give the same result and deallocating.
  check_equal_matrix(cpu_distances, gpu_distances);
  deallocate_gpu_distances(gpu_distances, n);
  return 0;
}
