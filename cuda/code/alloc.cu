
#include <cstdio>

/*
 * Initialize array values on the host.
 */

void init(int *a, int N)
{
  for (int i = 0; i < N; ++i) {
    a[i] = i;
  }
}

/*
 * Double elements in parallel on the GPU.
 */

__global__
void doubleElements(int *a, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    a[i] *= 2;
  }
}

/*
 * Check all elements have been doubled on the host.
 */

bool checkElementsAreDoubled(int *a, int N)
{
  for (int i = 0; i < N; ++i) {
    if (a[i] != i*2) {
      return false;
    }
  }
  return true;
}

int main()
{
  int N = 100;
  int *a;

  size_t size = N * sizeof(int);

  /*
   * Refactor this memory allocation to provide a pointer
   * 'a' that can be used on both the host and the device.
   */

  a = (int *)malloc(size);

  init(a, N);

  size_t threads_per_block = 10;
  size_t number_of_blocks = 10;

  /*
   * This launch will not work until the pointer 'a' is also
   * available to the device.
   */

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  /*
   * Refactor to free memory that has been allocated to be
   * accessed by both the host and the device.
   */

  free(a);
}
