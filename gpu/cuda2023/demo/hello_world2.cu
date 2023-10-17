// Copyright 2023 Pierre Talbot

#include <cstdio>

#define CUDIE(result) { \
  cudaError_t e = (result); \
  if (e != cudaSuccess) { \
    printf("%s:%d CUDA runtime error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }}

__host__ __device__ void print(const char* msg) {
  printf("%s\n", msg);
}

__global__ void hello_world() {
  print("world");
}

int main(int argc, char** argv) {
  print("hello");
  hello_world<<<1, 1>>>();
  CUDIE(cudaDeviceSynchronize())
  return 0;
}
