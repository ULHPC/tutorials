#include <cstdio>
#include <random>
#include <chrono>
#include <iostream>
#include <cooperative_groups.h>

#define CUDIE(result) { \
    cudaError_t e = (result); \
    if (e != cudaSuccess) { \
      printf("%s:%d CUDA runtime error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }}

__device__ __host__ void print_distances(int** distances, size_t n) {
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      printf("%2d ", distances[i][j]);
    }
    printf("\n");
  }
}

__device__ __host__ void floyd_warshall(int** distances, size_t n, int k) {
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      if(distances[i][j] > distances[i][k] + distances[k][j]) {
        distances[i][j] = distances[i][k] + distances[k][j];
      }
    }
  }
}

__device__ __host__ void floyd_warshall(int** distances, size_t n) {
  for(int k = 0; k < n; ++k) {
    floyd_warshall(distances, n, k);
  }
}

__global__ void shortest_path_one_thread(int** distances, size_t n) {
  // printf("%d\n", n);
  // print_distances(distances, n);
  floyd_warshall(distances, n);
  // printf("\n");
  // print_distances(distances, n);
}

std::vector<std::vector<int>> initialize_distances(int n) {
  std::vector<std::vector<int>> distances(n, std::vector<int>(n));
  // std::mt19937 m{std::random_device{}()};
  std::mt19937 m{0}; // fixed seed to ease debugging.
  std::uniform_int_distribution<int> dist{0, 99};
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      distances[i][j] = dist(m);
    }
  }
  return std::move(distances);
}

int** initialize_gpu_distances(const std::vector<std::vector<int>>& distances) {
  size_t n = distances.size();
  int** gpu_distances;
  cudaMallocManaged(&gpu_distances, sizeof(int*) * n);
  for(int i = 0; i < n; ++i) {
    cudaMallocManaged(&gpu_distances[i], sizeof(int) * n);
  }
  for(int i = 0; i < distances.size(); ++i) {
    for(int j = 0; j < distances[i].size(); ++j) {
      gpu_distances[i][j] = distances[i][j];
    }
  }
  return gpu_distances;
}

void deallocate_gpu_distances(int** gpu_distances, size_t n) {
  for(int i = 0; i < n; ++i) {
    cudaFree(gpu_distances[i]);
  }
  cudaFree(gpu_distances);
}

void floyd_warshall(std::vector<std::vector<int>>& distances) {
  size_t n = distances.size();
  for(int k = 0; k < n; ++k) {
    for(int i = 0; i < n; ++i) {
      for(int j = 0; j < n; ++j) {
        if(distances[i][j] > distances[i][k] + distances[k][j]) {
          distances[i][j] = distances[i][k] + distances[k][j];
        }
      }
    }
  }
}

__global__ void shortest_path_gpu(int** distances, size_t n, int k) {
  for(int i = blockIdx.x; i < n; i += gridDim.x) {
    for(int j = threadIdx.x; j < n; j += blockDim.x) {
      if(distances[i][j] > distances[i][k] + distances[k][j]) {
        distances[i][j] = distances[i][k] + distances[k][j];
      }
    }
  }
}

__global__ void shortest_path_gpu2(int** distances, size_t n) {
  auto grid = cooperative_groups::this_grid();
  for(int k = 0; k < n; ++k) {
    for(int i = blockIdx.x; i < n; i += gridDim.x) {
      for(int j = threadIdx.x; j < n; j += blockDim.x) {
        if(distances[i][j] > distances[i][k] + distances[k][j]) {
          distances[i][j] = distances[i][k] + distances[k][j];
        }
      }
    }
    grid.sync();
  }
}

int main(int argc, char** argv) {
  size_t n = 15000;
  auto distances = initialize_distances(n);
  auto gpu_distances = initialize_gpu_distances(distances);
  auto gpu_start = std::chrono::steady_clock::now();
  // for(int k = 0; k < n; ++k) {
  //   shortest_path_gpu<<<512, 512>>>(gpu_distances, n, k);
  // }

  void* args[] = {&gpu_distances, &n};
  dim3 dimBlock(160, 1, 1);
  dim3 dimGrid(270, 1, 1);
  CUDIE(cudaLaunchCooperativeKernel((void*)shortest_path_gpu2, dimGrid, dimBlock, args));

  CUDIE(cudaDeviceSynchronize());
  auto gpu_end = std::chrono::steady_clock::now();
  std::cout << "GPU: " <<  std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count() << " ms" << std::endl;
  auto cpu_start = std::chrono::steady_clock::now();
  floyd_warshall(distances);
  auto cpu_end = std::chrono::steady_clock::now();
  for(size_t i = 0; i < distances.size(); ++i) {
    for(size_t j = 0; j < distances.size(); ++j) {
      if(distances[i][j] != gpu_distances[i][j]) {
        printf("Found an error: %d != %d\n", distances[i][j], gpu_distances[i][j]);
      }
    }
  }
  deallocate_gpu_distances(gpu_distances, n);
  std::cout << "CPU: " <<  std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count() << " ms" << std::endl;
  return 0;
}
