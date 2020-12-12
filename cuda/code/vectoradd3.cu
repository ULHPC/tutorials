#include <cstdio>
#include "cuda.h"

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i) {
    a[i] = num;
  }
}

/*
 * Device kernel to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */

__global__
void initWithKernel(float num, float *a, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride) {
    a[i] = num;
  }
}

/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride) {
    result[i] = a[i] + b[i];
  }
}

/*
 * Host function to confirm values in `vector`. This function
 * assumes all values are the same `target` value.
 */

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++) {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  int deviceId;
  cudaGetDevice(&deviceId);
  printf("Device Id %d\n", deviceId);

  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a, *b, *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 512;
  numberOfBlocks = 80;

  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);

  initWithKernel<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
  initWithKernel<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
  initWithKernel<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
  
  cudaDeviceSynchronize();
  
  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

