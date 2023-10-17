// Copyright 2023 Pierre Talbot

#include "../utility.hpp"
#include <string>

__forceinline__ __device__ int dim2D(int x, int y, int n) {
  return x * n + y;
}

__global__ void floyd_warshall_gpu(int** d, size_t n) {
  // Copy the matrix into the shared memory.
  extern __shared__ int d2[];
  for(int i = 0; i < n; ++i) {
    for(int j = threadIdx.x; j < n; j += blockDim.x) {
      d2[dim2D(i, j, n)] = d[i][j];
    }
  }
  __syncthreads();
  // Compute on the shared memory.
  for(int k = 0; k < n; ++k) {
    for(int i = 0; i < n; ++i) {
      for(int j = threadIdx.x; j < n; j += blockDim.x) {
        if(d2[dim2D(i,j,n)] > d2[dim2D(i,k,n)] + d2[dim2D(k,j,n)]) {
          d2[dim2D(i,j,n)] = d2[dim2D(i,k,n)] + d2[dim2D(k,j,n)];
        }
      }
    }
    __syncthreads();
  }
  // Copy the matrix back to the global memory.
  for(int i = 0; i < n; ++i) {
    for(int j = threadIdx.x; j < n; j += blockDim.x) {
      d[i][j] = d2[dim2D(i, j, n)];
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
  if(argc != 3) {
    std::cout << "usage: " << argv[0] << " <matrix size> <threads-per-block>" << std::endl;
    exit(1);
  }
  size_t n = std::stoi(argv[1]);
  size_t threads_per_block = std::stoi(argv[2]);

  // I. Generate a random distance matrix of size N x N.
  std::vector<std::vector<int>> cpu_distances = initialize_distances(n);
  // Note that `std::vector` cannot be used on GPU, hence we transfer it into a simple `int**` array in managed memory.
  int** gpu_distances = initialize_gpu_distances(cpu_distances);

  // II. Running Floyd Warshall on CPU.
  long cpu_ms = benchmark_one_ms([&]{
    floyd_warshall_cpu(cpu_distances);
  });
  std::cout << "CPU: " << cpu_ms << " ms" << std::endl;

  // III. Running Floyd Warshall on GPU (single block of size `threads_per_block`).
  long gpu_ms = benchmark_one_ms([&]{
    floyd_warshall_gpu<<<1, threads_per_block, n * n * sizeof(int)>>>(gpu_distances, n);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU: " << gpu_ms << " ms" << std::endl;

  // IV. Verifying both give the same result and deallocating.
  check_equal_matrix(cpu_distances, gpu_distances);
  deallocate_gpu_distances(gpu_distances, n);
  return 0;
}
