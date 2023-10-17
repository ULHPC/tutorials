// Copyright 2023 Pierre Talbot

#include <cstdio>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>

#ifndef UTILITY_HPP
#define UTILITY_HPP

#define CUDIE(result) { \
  cudaError_t e = (result); \
  if (e != cudaSuccess) { \
    printf("%s:%d CUDA runtime error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }}

/** Initialize a matrix of size `n` with elements between 0 and 99. */
std::vector<std::vector<int>> initialize_distances(size_t n) {
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

/** Initialize an array of size `n` in managed memory. */
int* init_random_vector(size_t n) {
  int* v;
  CUDIE(cudaMallocManaged(&v, sizeof(int) * n));
  std::mt19937 m{std::random_device{}()};
  std::uniform_int_distribution<int> dist{1, std::numeric_limits<int>::max()};
  std::generate(v, v + n, [&dist, &m](){return dist(m);});
  return v;
}

/** Compare two matrices to ensure they are equal. */
void check_equal_matrix(const std::vector<std::vector<int>>& cpu_distances, int** gpu_distances) {
  for(size_t i = 0; i < cpu_distances.size(); ++i) {
    for(size_t j = 0; j < cpu_distances.size(); ++j) {
      if(cpu_distances[i][j] != gpu_distances[i][j]) {
        printf("Found an error: %d != %d\n", cpu_distances[i][j], gpu_distances[i][j]);
        exit(1);
      }
    }
  }
}

/** Copy a CPU matrix to the managed memory of the GPU. */
int** initialize_gpu_distances(const std::vector<std::vector<int>>& distances) {
  size_t n = distances.size();
  int** gpu_distances;
  CUDIE(cudaMallocManaged(&gpu_distances, sizeof(int*) * n));
  for(int i = 0; i < n; ++i) {
    CUDIE(cudaMallocManaged(&gpu_distances[i], sizeof(int) * n));
  }
  for(int i = 0; i < distances.size(); ++i) {
    for(int j = 0; j < distances[i].size(); ++j) {
      gpu_distances[i][j] = distances[i][j];
    }
  }
  return gpu_distances;
}

/** Deallocating the GPU matrix. */
void deallocate_gpu_distances(int** gpu_distances, size_t n) {
  for(int i = 0; i < n; ++i) {
    cudaFree(gpu_distances[i]);
  }
  cudaFree(gpu_distances);
}

/** Benchmarks the time taken by the function `f` by executing it 1 time first for warm-up, then 10 times in a row, then dividing the result by 10.
 * It returns the duration in milliseconds. */
template<class F>
long benchmark_ms(F&& f) {
  f(); // warm-up.
  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < 10; ++i) {
    f();
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 10;
}

template<class F>
long benchmark_one_ms(F&& f) {
  auto start = std::chrono::steady_clock::now();
  f();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 10;
}

#endif
