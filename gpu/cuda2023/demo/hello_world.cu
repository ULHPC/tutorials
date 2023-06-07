// Copyright 2023 Pierre Talbot

#include <cstdio>

__global__ void hello_world() {
  printf("hello world!\n");
}

int main(int argc, char** argv) {
  hello_world<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
