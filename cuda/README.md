[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/cuda/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/cuda/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/cuda/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Introduction to GPU programming with CUDA (C/C++)

     Copyright (c) 2013-2021 UL HPC Team <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/cuda/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/cuda/slides.pdf)


This tutorial will cover the following aspects of CUDA programming:

- Write, compile and run C/C++ programs that both call CPU functions and launch GPU kernels.
- Control parallel thread hierarchy using execution configuration.
- Allocate and free memory available to both CPUs and GPUs.
- Access memory on both GPU and CPU.
- Profile and improve the performance of your application.

Solutions to some of the exercises can be found in the [code sub-directory][3].

The tutorial is based on the Nvidia DLI course "Fundamentals of accelerated computing with CUDA C/C++".

More information can be obtained from [the guide][1].

[1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html "Programming guide"
[2]: https://docs.nvidia.com/cuda/cuda-runtime-api/index.html "Runtime API"
[3]: ./code "code"

--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc-docs.uni.lu/connect/access/).
In particular, recall that the `module` command **is not** available on the access frontends.

```bash
### Access to ULHPC cluster - here iris
(laptop)$> ssh iris-cluster
# /!\ Advanced (but recommended) best-practice:
#    always work within an GNU Screen session named with 'screen -S <topic>' (Adapt accordingly)
# IIF not yet done, copy ULHPC .screenrc in your home
(access)$> cp /etc/dotfiles.d/screen/.screenrc ~/
```

Now you'll need to pull the latest changes in your working copy of the [ULHPC/tutorials](https://github.com/ULHPC/tutorials) you should have cloned in `~/git/github.com/ULHPC/tutorials` (see ["preliminaries" tutorial](../../preliminaries/))

``` bash
(access)$> cd ~/git/github.com/ULHPC/tutorials
(access)$> git pull
```


## Access to a GPU-equipped node of the Iris cluster

Reserve a node with one GPU for interactive development, load the necessary modules, and save them for a quick restore.

As usual, more information can be found in the [documentation][4].

[4]: https://hpc-docs.uni.lu/jobs/gpu/

```bash
### Have an interactive GPU job
# ... either directly
(access)$> si-gpu
# ... or using the HPC School reservation 'hpcschool-gpu' if needed  - use 'sinfo -T' to check if active and its name
# (access)$> si-gpu --reservation=hpcschool-gpu
$ nvidia-smi
$ nvcc  # ?
```

Driver is loaded, but we still need to load the CUDA development kit.

```bash
$ module av cuda  # av versus spider
----------------------------------------------------------- /opt/apps/resif/data/stable/default/modules/all ------------------------------------------------------------
   bio/GROMACS/2019.2-fosscuda-2019a                    mpi/impi/2018.4.274-iccifortcuda-2019a
   bio/GROMACS/2019.2-intelcuda-2019a                   numlib/FFTW/3.3.8-intelcuda-2019a
   compiler/Clang/8.0.0-GCCcore-8.2.0-CUDA-10.1.105     numlib/cuDNN/7.4.2.24-gcccuda-2019a
   data/h5py/2.9.0-fosscuda-2019a                       numlib/cuDNN/7.6.4.38-gcccuda-2019a                       (D)
   data/h5py/2.9.0-intelcuda-2019a                      system/CUDA/9.2.148.1
   devel/PyTorch/1.2.0-fosscuda-2019a-Python-3.7.2      system/CUDA/10.0.130
   lang/SciPy-bundle/2019.03-fosscuda-2019a             system/CUDA/10.1.105-GCC-8.2.0-2.31.1
   lang/SciPy-bundle/2019.03-intelcuda-2019a            system/CUDA/10.1.105-iccifort-2019.1.144-GCC-8.2.0-2.31.1
   lib/NCCL/2.4.7-gcccuda-2019a                         system/CUDA/10.1.105
   lib/TensorFlow/1.13.1-fosscuda-2019a-Python-3.7.2    system/CUDA/10.1.243                                      (D)
   lib/TensorRT/6.0.1.5-fosscuda-2019a-Python-3.7.2     system/CUDA/10.2.89
   lib/libgpuarray/0.7.6-fosscuda-2019a                 toolchain/fosscuda/2019a
   math/Keras/2.2.4-fosscuda-2019a-Python-3.7.2         toolchain/gcccuda/2019a
   math/Theano/1.0.4-fosscuda-2019a                     toolchain/iccifortcuda/2019a
   math/magma/2.5.1-fosscuda-2019a                      toolchain/intelcuda/2019a
   mpi/OpenMPI/3.1.4-gcccuda-2019a                      tools/Horovod/0.16.3-fosscuda-2019a-Python-3.7.2

  Where:
   D:  Default Module

Use "module spider" to find all possible modules.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".

$ module av gcc cuda
$ module load compiler/GCC system/CUDA  # defaults are fine
$ ml
$ module save cuda  # save our environment
$ module purge
$ module restore cuda
```

In case there is not enough GPU cards available, you can submit passive jobs, using `sbatch`.
Below is an example `sbatch` file, to remote compile, run and profile a source file:

```bash
#!/bin/bash -l
#SBATCH --job-name="GPU build"
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

if [ -z "$1" ]
then
	echo "Missing required source (.cu), and optional execution arguments."
	exit
fi

src=${1}
exe=$(basename ${1/cu/out})
ptx=$(basename ${1/cu/ptx})
prf=$(basename ${1/cu/prof})
shift
args=$*

# after the module profile is saved (see above)
module restore cuda

# compile
srun nvcc -arch=compute_70 -o ./$exe $src
# save ptx
srun nvcc -ptx -arch=compute_70 -o ./$ptx $src
# execute
srun ./$exe $args
# profile
srun nvprof --log-file ./$prf ./$exe $args
echo "file: $prf"
cat ./$prf
```

## Writing application for the GPU

CUDA provides extensions for many common programming languages, in the case of this tutorial, C/C++.
There are several API available for GPU programming, with either specialization, or abstraction.
The main API is the *CUDA Runtime*.
Another, lower level API, is *CUDA Driver*, which also offers more customization options.
Example of other APIs, built on top of the CUDA Runtime, are Thrust, NCCL.

### Hello World

Below is a example CUDA `.cu` program (`.cu` is the *required* file extension for CUDA-accelerated programs).
It contains two functions, the first which will run on the CPU, the second which will run on the GPU.

```cpp
#include <cstdio>
#include "cuda.h"

void CPUFunction()
{
  printf("hello from the Cpu.\n");
}

__global__
void GPUFunction()
{
  printf("hello from the Gpu.\n");
}

int main()
{
  CPUFunction();

  GPUFunction<<<1, 1>>>();

  cudaDeviceSynchronize();

  return EXIT_SUCCESS;
}
```
Here are some important lines to highlight, as well as some other common terms used in accelerated computing:

```cpp
__global__
void GPUFunction()
```

The `__global__` keyword indicates that the following function will run on the GPU, and can be invoked globally, which means either by the CPU or GPU.
Often, code executed on the CPU is referred to as *host* code, and code running on the GPU is referred to as *device* code.
Notice the return type `void`.
It is required that functions defined with the `__global__` keyword return type `void`.

```cpp
GPUFunction<<<1, 1>>>();
```

Typically, when calling a function to run on the GPU, we call this function a kernel, which is launched.
When launching a kernel, we must provide an execution configuration, which is done by using the `<<< ... >>>` syntax just prior to passing the kernel any expected arguments.
At a high level, execution configuration allows programmers to specify the thread hierarchy for a kernel launch, which defines the number of thread groupings (called blocks), as well as how many threads to execute in each block.

The execution configuration allows programmers to specify details about launching the kernel to run in parallel on multiple GPU threads.
More precisely, the execution configuration allows programmers to specifiy how many groups of threads - called thread blocks, or just blocks - and how many threads they would like each thread block to contain.
The simplest syntax for this is: (there are 2 othe parameters available)
```cpp
<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>
```
The kernel code is executed by every thread in every thread block configured when the kernel is launched.

This illustrates the data parallel model of CUDA. The same function is executed on all threads.

```cpp
cudaDeviceSynchronize();
```
Unlike much C/C++ code, launching kernels is asynchronous: the CPU code will continue to execute without waiting for the kernel launch to complete.
A call to ``cudaDeviceSynchronize`, a function provided by the CUDA runtime, will cause the host (CPU) code to wait until the device (GPU) code completes, and only then resume execution on the CPU.

### Compiling and running CUDA code

This section contains details about the *nvcc* command you issue to compile and run your `.cu` program.

The CUDA platform ships with the NVIDIA CUDA Compiler `nvcc`, which can compile CUDA accelerated applications, both the host, and the device code they contain.
After completing the lab, For anyone interested in a deeper dive into `nvcc`, start with the documentation (`nvcc --help`).

Compiling and executing `some-CUDA.cu` source file:
```bash
$nvcc -arch=sm_70 -o out some-CUDA.cu -run
```

`nvcc` is the command line command for using the nvcc compiler.
`some-CUDA.cu` is passed as the source file to compile.
The `o` flag is used to specify the output file for the compiled program.
The `arch` flag indicates for which architecture the files must be compiled.
For the present case `sm_70` will serve to compile specifically for the Volta GPUs.
This will be further explained in a following section.
As a matter of convenience, providing the `-run` flag will execute the successfully compiled binary.

`nvcc` parses the C++ language (it used to be C).

### Practice: Launch parallel kernels

The following program currently makes a function call that prints a message, but it is incorrect.

Fix and refactor the code such that `helloGPU` kernel to execute in parallel on 5 threads, all executing in a single thread block.
You should see the output message printed 5 times after compiling and running the code.

Refactor the `helloGPU` kernel again, this time to execute in parallel inside 5 thread blocks, each containing 5 threads.
You should see the output message printed 25 times now after compiling and running.

```cpp
/*
 * FIXME
 * (hello.cu)
 */

void helloCPU()
{
  std::cout<<"Hello from Cpu.\n";
}

void helloGPU()
{
  printf("Hello also from Gpu.\n");
}

int main()
{

  helloCPU();
  helloGPU();

  return EXIT_SUCCESS;
}

```

### Compiling and running CUDA code, continued

The following compilation command works:
```bash
$ nvcc some-CUDA.cu
```
However, there is a potential problem with this code.
Cuda uses a two stage compilation process, to PTX, and to binary.

To produce the PTX for the cuda kernel, use:
```bash
$ nvcc -ptx -o out.ptx some-CUDA.cu
```
Brief inspection of the generated PTX file reports a *target* real platform of `sm_30`.
```
// Based on LLVM 3.4svn
//

.version 6.4
.target sm_30
.address_size 64
```
The Volta GPU implements `sm_70`. So, we could be missing some features from the GPU.

To specify an instruction set, use the `-arch` option:
```bash
$ nvcc -o out.ptx -ptx -arch=compute_70 some-CUDA.cu
```
To produce an executable, instead of just the PTX code, and specify an instruction set, use the `-arch` option, for example:
```bash
$ nvcc -o out -arch=compute_70 some-CUDA.cu
```
This actually produces an executable that embeds the kernels' code as PTX, of the specified instruction set.
The PTX code will then be JIT compiled when executed, matching the real GPU instruction set.
To see this, search for the target PTX instruction in the executable:
```bash
$ strings out | grep target
```

The `code` option specifies what the executable contains.
The following nvcc options specify that the executables contains the binary code for the real GPU `sm_70`.
```bash
$ nvcc -o out -arch=compute_70 -code=sm_70 some-CUDA.cu
```
The following nvcc options specify that the executables contains the binary code for the real GPU `sm_70`, *and* the PTX code for the `sm_70`.
```bash
$ nvcc -o out -arch=compute_70 -code=sm_70,compute_70 some-CUDA.cu
```
To observe the difference, search for the target PTX command, in both commands:
```bash
$ strings out | grep target
```
Actually, the first compilation instruction:
```bash
$ nvcc -o out -arch=sm_70 some-CUDA.cu
```
is a shorthand for the full command:
```bash
$ nvcc -o out -arch=compute_70 -code=sm_70,compute_70 some-CUDA.cu
```

In summary, if you want to package the PTX to allow for JIT compilation across different real GPU:
```bash
$ nvcc -o out -arch=compute_70 some-CUDA.cu  # or sm_70
```

### Error handling

It is strongly recommended to check for errors when calling CUDA API.a

```cpp
cudaError_t rc;  # cudaSuccess => ok
rc = cudaDeviceSynchronize();
printf("%s\n", cudaGetErrorString(rc));

// for asynchronous calls:
rc = cudaGetLastError();  # call after synchronization for post-launch kernel errors
```

For convenience, the following macros can help:
```cpp
#define CUDIE(result) { \
        cudaError_t e = (result); \
        if (e != cudaSuccess) { \
                std::cerr << __FILE__ << ":" << __LINE__; \
                std::cerr << " CUDA runtime error: " << cudaGetErrorString(e) << '\n'; \
                exit((int)e); \
        }}

#define CUDIE0() CUDIE(cudaGetLastError())
```

For example, try compiling and executing code on iris (Volta GPU) with `-arch=sm_75` (or `compute_75`).


## CUDA thread hierarchy

Each thread will execute the same kernel function.
Therefore, some coordination mechanism is needed for each thread to work on different memory locations.
A mechanism involves builtin constants, that are used to uniquely identify threads of an execution configuration.
Other aspects involve thread synchronization, streams, events, cooperative groups (not covered here besides the host-side synchronization function), see the [reference][2].

### Thread and block indices
Each thread is given an index within its thread block, starting at 0.
Additionally, each block is given an index, starting at 0.
Just as threads are grouped into thread blocks, blocks are grouped into a grid, which is the highest entity in the CUDA thread hierarchy.
In summary, CUDA kernels are executed in a grid of 1 or more blocks, with each block containing the same number of 1 or more threads.

CUDA kernels have access to special variables identifying both the index of the thread (within the block) that is executing the kernel, and, the index of the block (within the grid) that the thread is within.
These variables are `threadIdx.x` and `blockIdx.x` respectively.

The `.x` suggests that there more dimensions to these variables, they can be up to 3 dimensions.
We will only see examples with one dimension here.
Refer to the [programming guide][1] for more information.

Within a block, the thread ID is the same as the threadIdx.x.
There is a maximum of 1024 threads allowed per block.
Within a one-dimensional grid, the block ID is the same as the blockIdx.x.
Together, the number of blocks and number of threads allow to exceed the 1024 threads limit.

### Exercise: use specific thread and block indices

The program below contains a working kernel that is not printing a success message.
Edit the source code to update the execution configuration so that the success message will print.
```cpp
/*
 * FIXME
 * (indices.cu)
 */

#include <cstdio>

void printif()
{
  if (threadIdx.x == 1023 && blockIdx.x == 255) {
    printf("Success!\n");
  }
}

int main()
{
  /*
   * Update the execution configuration so that the kernel
   * will print `"Success!"`.
   */

  printif<<<1, 1>>>();
}
```

### Accelerating `for` loops

For loops in CPU applications can sometimes be accelerated: rather than run each iteration of the loop sequentially, each iteration of the loop can be run in parallel in its own thread.
Consider the following for loop, and notice that it controls how many times the loop will execute, as well as defining what will happen for each iteration of the loop:

```cpp
int N = 2<<10;
for (int i = 0; i < N; ++i) {
  printf("%d\n", i);
}
```
In order to parallelize this loop, 2 steps must be taken:

- A kernel must do the work of a single iteration of the loop.
- Because the kernel will ignore other running kernels, the execution configuration must ensure that the kernel executes the correct number of times, for example, the number of times the loop would have iterated.

### Exercise: Accelerating a For Loop with a Single Block of Threads

Currently, the loop function runs a for loop that will serially print the numbers 0 through 9.
Modify the loop function to be a CUDA kernel which will launch to execute N iterations in parallel.
After successfully refactoring, the numbers 0 through 9 should still be printed.
```cpp
/*
 * FIXME
 * (loop.cu)
 * Correct, and refactor 'loop' to be a CUDA Kernel.
 * The new kernel should only do the work
 * of 1 iteration of the original loop.
 */

#include <cstdio>

void loop(int N)
{
  for (int i = 0; i < N; ++i) {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{
  /*
   * When refactoring 'loop' to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * Use 1 block of threads.
   */

  int N = 10;
  loop(N);
}
```

### Using block dimensions for more parallelization

CUDA Kernels have access to another special variable that gives the number of threads in a block: `blockDim.x`.
Using this variable, in conjunction with `blockIdx.x` and `threadIdx.x`, increased parallelization can be accomplished by organizing parallel execution accross multiple blocks of multiple threads with the idiomatic expression `threadIdx.x + blockIdx.x * blockDim.x`.

### Exercise: accelerating a for loop with multiple blocks of threads

Make further modifications to the previous exercise, but with a execution configuration that launches at least 2 blocks.
After successfully refactoring, the numbers 0 through 9 should still be printed.

```cpp
/*
 * FIXME
 * (loop2.cu)
 * Fix and refactor 'loop' to be a CUDA Kernel, launched with 2 or more blocks
 * The new kernel should only do the work of 1 iteration of the original loop.
 */

#include <cstdio>

void loop(int N)
{
  for (int i = 0; i < N; ++i) {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{
  /*
   * When refactoring 'loop' to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, be sure to use more than 1 block in
   * the execution configuration.
   */

  int N = 10;
  loop(N);
}
```

## Memory allocation

### Allocating memory to be accessed on the GPU and the CPU

For any meaningful work to be done, we need to access memory.
The GPU has a distinct memory from the CPU, which requires data transfers to and from CPU.
However, recent versions of CUDA (version 6 and later) have simplified memory allocation that is available to both the CPU host and any number of GPU devices, and while there are many intermediate and advanced techniques for memory management that will support the most optimal performance in accelerated applications, the most basic CUDA memory management technique we will now cover supports good performance gains over CPU-only applications.
The simplified memory allocation is called [Unified Memory System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd), and has many implications on multi-GPU systems.

To allocate and free memory, and obtain a pointer that can be referenced in both host and device code, replace calls to malloc and free with `cudaMallocManaged` and `cudaFree` as in the following example:

```cpp
//--- CPU-only ---

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
a = (int *)malloc(size);

// Use 'a' in CPU-only program.

free(a);

//--- GPU/CPU ---

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
// Note the address of 'a' is passed as first argument.
cudaMallocManaged(&a, size);

// Use 'a' on the CPU and/or on any GPU in the accelerated system.

cudaFree(a);
```

For completeness, the alternative to unified memory is to:

- manually allocate memory on a device
- copy data from host to device memory.

This can be done with:
```cpp
float* d_C;
cudaMalloc(&d_C, size);
// Copy vectors from host memory to device memory
cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);  // to mimick d_C = h_C;
cudaFree(d_C);
```

### Exercise: array manipulation on both the host and device

The source code below allocates an array, initializes it with integer values on the host, attempts to double each of these values in parallel on the GPU, and then confirms whether or not the doubling operations were successful, on the host.
Currently the program will not work: it is attempting to interact on both the host and the device with an array at pointer `a`, but has only allocated the array (using malloc) to be accessible on the host.
Refactor the application to meet the following conditions:

- `a` should be available to both host and device code.
- The memory at a should be correctly freed.


```cpp
/*
 * FIXME
 * (alloc.cu)
 */

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
```

A solution can be found in the next exercise.

### Data sets larger than the grid

Either by choice, often to create the most performant execution configuration, or out of necessity, the number of threads in a grid may be smaller than the size of a data set.
Consider an array with 1000 elements, and a grid with 250 threads.
Here, each thread in the grid will need to be used 4 times.
One common method to do this is to use a *grid-stride* loop within the kernel.

In a grid-stride loop, each thread will calculate its unique index within the grid using `tid+bid*bdim`, perform its operation on the element at that index within the array, and then, add to its index the number of threads in the grid and repeat, until it is out of range of the array.
For example, for a 500 element array and a 250 thread grid, the thread with index 20 in the grid would:

- Perform its operation on element 20 of the 500 element array.
- Increment its index by 250, the size of the grid, resulting in 270.
- Perform its operation on element 270 of the 500 element array.
- Increment its index by 250, the size of the grid, resulting in 520.
- Because 520 is now out of range for the array, the thread will stop its work.

CUDA provides a special builtin variable giving the number of blocks in a grid: `gridDim.x`.
Calculating the total number of threads in a grid then is simply the number of blocks in a grid multiplied by the number of threads in each block, `gridDim.x * blockDim.x`.
With this in mind, here is a verbose example of a grid-stride loop within a kernel:
```cpp
__global__
void kernel(int *a, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    // do work on a[i];
  }
}
```

### Exercise: use a grid-stride loop to manipulate an array larger than the grid

Refactor the previous code to use a grid-stride loop in the `doubleElements` kernel, in order that the grid, which is smaller than N, can reuse threads to cover every element in the array.
The program will print whether or not every element in the array has been doubled, currently the program accurately prints `FALSE`.
```cpp
/*
 * FIXME
 * (loop-stride.cu)
 * using strides
 */

#include <cstdio>

void init(int *a, int N)
{
  for (int i = 0; i < N; ++i) {
    a[i] = i;
  }
}

/*
 * In the current application, 'N' is larger than the grid.
 * Refactor this kernel to use a grid-stride loop in order that
 * each parallel thread work on more than one element of the array.
 */

__global__
void doubleElements(int *a, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    a[i] *= 2;
  }
}

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
  /*
   * 'N' is greater than the size of the grid (see below).
   */

  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  /*
   * The size of this grid is 256*32 = 8192.
   */

  size_t threads_per_block = 256;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
```
One solution is in file `sol-array-stride.cu`.

### Handling block configuration mismatches to number of needed threads (minor)

It may be the case that an execution configuration cannot be expressed to create the exact number of threads needed for parallelizing a loop.

A common example has to do with the desire to choose optimal block sizes.
For example, due to GPU hardware traits, blocks that contain a number of threads that are a multiple of 32 are often desirable for performance benefits.
Assuming that we wanted to launch blocks each containing 256 threads (a multiple of 32), and needed to run 1000 parallel tasks, then there is no number of blocks that would produce an exact total of 1000 threads in the grid, since there is no integer value 32 can be multiplied by to equal exactly 1000.

This scenario can be easily addressed in the following way:

Write an execution configuration that creates more threads than necessary to perform the allotted work.
Pass a value as an argument into the kernel (N) that represents to the total size of the data set to be processed, or the total threads that are needed to complete the work.
After calculating the thread's index within the grid (using `tid+bid*bdim`), check that this index does not exceed N, and only perform the pertinent work of the kernel if it does not.
Here is an example of an idiomatic way to write an execution configuration when both N and the number of threads in a block are known, and an exact match between the number of threads in the grid and N cannot be guaranteed.
It ensures that there are always at least as many threads as needed for N, and only 1 additional block's worth of threads extra, at most:

```cpp
// Assume 'N' is known
int N = 100000;

// Assume we have a desire to set 'threads_per_block' exactly to '256'
size_t threads_per_block = 256;

// Ensure there are at least 'N' threads in the grid, but only 1 block's worth extra
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

some_kernel<<<number_of_blocks, threads_per_block>>>(N);
```
Because the execution configuration above results in more threads in the grid than N, care will need to be taken inside of the some_kernel definition so that some_kernel does not attempt to access out of range data elements, when being executed by one of the "extra" threads:
```cpp
__global__
some_kernel(int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N) // Check to make sure 'idx' maps to some value within 'N'
  {
    // Only do work if it does
  }
}
```


### Shared memory

On first approximation, there are 3 memories available to your kernel code:

- global memory, large but slow,
- block shared memory, small but fast,
- registers.

Within a kernel code:

- automatic variable are stored in registers
- `__shared__` specifier indicates the block memory, faster than global memory (limited, 96KB).

Shared memory can be declared outside the kernel code with:
```cpp
extern __shared__ float shared[];
```
However, then the size of the shared memory must be declared in the kernel configuration:
```cpp
kernel<<<2, 32, 16*sizeof(int)>>>();
```

```cpp
extern __shared__ int a_blk[];

__global__ void doubleElements(int *a, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<N) {
          a_blk[threadIdx.x] = a[i];
          a_blk[threadIdx.x] *= 2;  // or something more complex
          a[i] = a_blk[threadIdx.x];
  }
}
// ...
doubleElements<<<10, 16, 16*sizeof(*a)>>>(a, N);
cudaDeviceSynchronize();
// ...
```

Note: `__syncthreads()` can be used to synchronize threads of a thread block: waits for all shared and global access to be visible from all threads (in a block).


## Performance considerations

In this section, we will investigate how to improve the performance (runtime) of a CUDA application.

We'll be looking at:

- Measuring the performance.
- Features that affect peformance: execution configuration and memory management.

### Preparation

The starting point for all experiments is a simple vector addition program `vectoradd.cu`, which can be found in the [samples sub-directory][3].
The file is included here:
```cpp
#include <stdio.h>
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
 * Device kernel stores into 'result' the sum of each
 * same-indexed value of 'a' and 'b'.
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
 * Host function to confirm values in 'vector'. This function
 * assumes all values are the same 'target' value.
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
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a, *b, *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 1;
  numberOfBlocks = 1;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  cudaDeviceSynchronize();

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
```
In case you need to setup the environment, issue the same interactive reservation as before:

```bash
> si-gpu
$ module r cuda  # restores our saved 'cuda' modules
```
Or use the `sbatch` script presented at the beginning of this tutorial.

### Profiling

We'll be using `nvprof` for this tutorial ([documentation][4]).

[4]: https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview "nvprof documentation"

To profile your application simply:
```bash
$ nvprof ./a.out  # you can also add --log-file prof
```
The default output includes 2 sections:

- one related to kernel and API calls
- another related to memory.

### Execution configuration

First, we look at the top part of the profiling result, related to function calls.

After profiling the application, can we answer the following questions:

- What was the name of the only CUDA kernel called in this application?
- How many times did this kernel run?
- How long did it take this kernel to run? Record this time somewhere: you will be optimizing this application and will want to know how much faster you can make it.

Experiment with different values for the number of threads, keeping only 1 block.

Experiment with different values for both the number of threads and number of blocks.

The GPUs that CUDA applications run on have processing units called streaming multiprocessors, or SMs.
During kernel execution, blocks of threads are given to SMs to execute.
In order to support the GPU's ability to perform as many parallel operations as possible, performance often improves by choosing a grid size that has a number of blocks that is a multiple of the number of SMs on a given GPU.

Additionally, SMs create, manage, schedule, and execute groupings of 32 threads from within a block called warps. i
A more in depth coverage of SMs and warps is beyond the scope of this course, however, it is important to know that performance gains can also be had by choosing a block size that has a number of threads that is a multiple of 32.

### Unified Memory details

Now, we turn to the memory related performance counters.

You have been allocating memory intended for use either by host or device code with cudaMallocManaged and up until now have enjoyed the benefits of this method - automatic memory migration, ease of programming - without diving into the details of how the Unified Memory (UM) allocated by cudaMallocManaged actual works.
`nvprof` provides details about UM management in accelerated applications, and using this information, in conjunction with a more-detailed understanding of how UM works, provides additional opportunities to optimize accelerated applications.

When Unified Memory is allocated, the memory is not resident yet on either the host or the device.
When either the host or device attempts to access the memory, a page fault will occur, at which point the host or device will migrate the needed data in batches.
Similarly, at any point when the CPU, or any GPU in the accelerated system, attempts to access memory not yet resident on it, page faults will occur and trigger its migration.

The ability to page fault and migrate memory on demand is helpful for ease of development in your accelerated applications.
Additionally, when working with data that exhibits sparse access patterns, for example when it is impossible to know which data will be required to be worked on until the application actually runs, and for scenarios when data might be accessed by multiple GPU devices in an accelerated system with multiple GPUs, on-demand memory migration is remarkably beneficial.
There are times - for example when data needs are known prior to runtime, and large contiguous blocks of memory are required - when the overhead of page faulting and migrating data on demand incurs an overhead cost that would be better avoided.

We'll be working with the sample code below (also in `um.cu`):

```cpp
__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  /*
   * Conduct experiments to learn more about the behavior of
   * `cudaMallocManaged`.
   *
   * What happens when unified memory is accessed only by the GPU?
   * What happens when unified memory is accessed only by the CPU?
   * What happens when unified memory is accessed first by the GPU then the CPU?
   * What happens when unified memory is accessed first by the CPU then the GPU?
   *
   * Hypothesize about UM behavior, page faulting specificially, before each
   * experiement, and then verify by running `nvprof`.
   */

  cudaFree(a);
}
```
For each of the 4 questions below, first hypothesize about what kind of page faulting should happen, then, edit the program to create a scenario that will allow you to test your hypothesis.

Be sure to record your hypotheses, as well as the results, obtained from `nvprof` output, specifically CPU and GPU page faults, for each of the 4 experiments you are conducting.

- What happens when unified memory is accessed only by the CPU?
- What happens when unified memory is accessed only by the GPU?
- What happens when unified memory is accessed first by the CPU then the GPU?
- What happens when unified memory is accessed first by the GPU then the CPU?

### GPU-side initialization

With this in mind, refactor your `vectoradd.cu` program to instead be a CUDA kernel, initializing the allocated vector in parallel on the GPU.
After successfully compiling and running the refactored application, but before profiling it, hypothesize about the following:

- How do you expect the refactor to affect UM page-fault behavior?
- How do you expect the refactor to affect the reported run time of addVectorsInto?
- Once again, record the results.

File `vectoradd2.cu` is one implementation.

### Asynchronous memory prefetching

A powerful technique to reduce the overhead of page faulting and on-demand memory migrations, both in host-to-device and device-to-host memory transfers, is called asynchronous memory prefetching.
Using this technique allows programmers to asynchronously migrate unified memory (UM) to any CPU or GPU device in the system, in the background, prior to its use by application code.
By doing this, GPU kernels and CPU function performance can be increased on account of reduced page fault and on-demand data migration overhead.

Prefetching also tends to migrate data in larger chunks, and therefore fewer trips, than on-demand migration.
This makes it an excellent fit when data access needs are known before runtime, and when data access patterns are not sparse.

CUDA Makes asynchronously prefetching managed memory to either a GPU device or the CPU easy with its `cudaMemPrefetchAsync` function.
Here is an example of using it to both prefetch data to the currently active GPU device, and then, to the CPU:
```cpp
int deviceId;
cudaGetDevice(&deviceId);                                         // The ID of the currently active GPU device.

cudaMemPrefetchAsync(pointerToSomeUMData, size, deviceId);        // Prefetch to GPU device.
cudaMemPrefetchAsync(pointerToSomeUMData, size, cudaCpuDeviceId); // Prefetch to host. 'cudaCpuDeviceId' is a
                                                                  // built-in CUDA variable.
```

At this point, your `vectoradd.cu` program should be (a) launching a CUDA kernel to add 2 vectors into a third solution vector, all which are allocated with cudaMallocManaged, and (b) initializing each of the 3 vectors in parallel in a CUDA kernel.

Conduct 3 (or 4) experiments using `cudaMemPrefetchAsync` inside of your `vectoradd.cu` application to understand its impact on page-faulting and memory migration.

- What happens when you prefetch one of the initialized vectors to the device?
- What happens when you prefetch two of the initialized vectors to the device?
- What happens when you prefetch all three of the initialized vectors to the device?

Hypothesize about UM behavior, page faulting specificially, as well as the impact on the reported run time of the initialization kernel, before each experiement, and then verify by running `nvprof`.
