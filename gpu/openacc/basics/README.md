[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/gpu/openacc/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/gpu/openacc/basics) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/gpu/openacc/basics/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Introduction to OpenACC Programming Model (C/C++ and Fortran)

     Copyright (c) E. Krishnasamy, 2013-2023 UL HPC Team <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/gpu/openacc/basics/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/gpu/openacc/basics/slides.pdf)

# Objectives

* Understanding the [OpenACC programming model](https://www.openacc.org/)
* How to use some of the directives from OpenACC to parallelize the code
      - compute constructs, loop constructs, data clauses
* Implementing OpenACC parallel strategy in [C/C++](https://www.cprogramming.com/) and [FORTRAN](https://fortran-lang.org/) programming languages
* Simple [mathematical examples](https://www.netlib.org/blas/) to support and understand the OpenACC programming model
* Finally, show you how to run these examples using [Iris cluster (ULHPC)](https://hpc-docs.uni.lu/systems/iris/)
          - both [interactively](https://hpc-docs.uni.lu/jobs/interactive/) and using a [batch job script](https://hpc-docs.uni.lu/jobs/gpu/)

### Prerequisite:
* C/C++ and/or FORTRAN languages
* OpenMP or some basic parallel programming concept (advantage not necessary)


### Important: PRACE MOOC

> ***NOTE***: this lecture is limited to just 45 min; it only a covers very basic tutorial about OpenACC.
  To know more about (from basic to advanced) CUDA programming and OpenACC programming model,
  please refer to __PRACE MOOC__ ***GPU Programming for Scientific Computing and Beyond - Dr. Ezhilmathi Krishnasamy and Prof. Pascal Bouvry***


--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc-docs.uni.lu/connect/access/).
In particular, recall that the `module` command **is not** available on the access frontends. **For all tests and compilation, you MUST work on a computing node**

Now you'll need to pull the latest changes in your working copy of the [ULHPC/tutorials](https://github.com/ULHPC/tutorials) you should have cloned in `~/git/github.com/ULHPC/tutorials` (see ["preliminaries" tutorial](../../preliminaries/))

```bash
(access)$ cd ~/git/github.com/ULHPC/tutorials
(access)$ git pull
```

Now **configure a dedicated directory `~/tutorials/openacc` for this session**

```bash
# return to your home
(access)$ mkdir -p ~/tutorials/openacc
(access)$ cd ~/tutorials/openacc
# create a symbolic link to the top reference material
(access)$ ln -s ~/git/github.com/ULHPC/tutorials/gpu/openacc/basics ref.d # Symlink to the reference tutorial material
# copy / synchronize a copy of the exercises
(access)$ rsync -avzu ref.d/exercises .   # DO NOT forget the trailing .
```

**Advanced users** (_eventually_ yet __strongly__ recommended), create a [Tmux](https://github.com/tmux/tmux/wiki) session (see [Tmux cheat sheet](https://tmuxcheatsheet.com/) and [tutorial](https://www.howtogeek.com/671422/how-to-use-tmux-on-linux-and-why-its-better-than-screen/)) or [GNU Screen](http://www.gnu.org/software/screen/) session you can recover later. See also ["Getting Started" tutorial ](../../beginners/).

#### Connect to Iris cluster and get an interactive GPU job

See also [GPU jobs documentation](https://hpc-docs.uni.lu/jobs/gpu/).

```bash
$ ssh iris-cluster   # The only cluster featuring GPU
$ si-gpu -G 1 --ntasks-per-node 1 -c 7 -t 00:30:00   # (eventually) --reservation=hpcschool-gpu
```

#### Load the OpenACC compiler (Nvidia HPC SDK)

```
$> module spider nvhpc
$> module load compiler/NVHPC/21.2
```

# Difference between CPU and GPU
#### CPU vs GPU

* A CPU frequency is higher compared to a GPU.
* But a GPU can run many threads in parallel compared to a CPU.
* On a GPU, the cores are grouped and called "Streaming Multiprocessor - SM".
* Even the Nvidia GPU has a "Tensor Process Unit - TPU" to handle the AI/ML
  computations in an optimized way.
* GPUs are based on the "Single Instruction Multiple Threads".
* Threads are executed in a group on the GPU; typically they have 32 threads.
  This is called "warps" on the Nvidia GPU and "wavefronts" on the AMD GPU.


#### CPU vs GPU
![](./Nvidia-cpu-gpu.png)


#### Thread Hierarchy
![](./grid-of-thread-blocks.png)
![](./memory-hierarchy.png)


#### How GPUs are used for computations

* Step 1: application preparation, initialize the memories on both CPU and GPU
* Step 2: transfer the data to a GPU
* Step 3: do the computation on a GPU
* Step 4: transfer the data back to a CPU
* Step 5: finalize the application and delete the memories on both CPU and GPU

![](./Nvidia-cpu-gpu-parallel.png)

# OpenACC
#### Few points about OpenACC

* OpenACC is not GPU programming
* OpenACC is expressing the parallelism in your existing code
* OpenACC can be used in both Nvidia and AMD GPUs
* “OpenACC will enable programmers to easily develop portable applications that maximize
  the performance and power efficiency benefits of the hybrid CPU/GPU architecture of
  Titan.”
      - Buddy Bland, Titan Project Director, Oak Ridge National Lab
* “OpenACC is a technically impressive initiative brought together by members of the
  OpenMP Working Group on Accelerators, as well as many others. We look forward to
  releasing a version of this proposal in the next release of OpenMP.”
      - Michael Wong, CEO of OpenMP Directives Board


#### Ways to accelerate applications on the GPU

![](./OpenACC-Three-Ways.png)


#### Ways to accelerate applications on the GPU

* Libraries: easy to use with very limited knowledge with GPU programming
      - cuBLAS, cuFFT, CUDA Math Library, etc.
* Directive based programming model: will accelerate the application
  with using directives in the existing code
      - OpenACC and OpenMP (might be applicable in the future)
* Programming languages: low level programming languages that will
  further optimize the application on the accelerator
      - CUDA, OpenCL, etc.


#### Compilers and directives (only a few of them are listed here)

* OpenACC is supported by the Nvidia, [PGI](https://www.pgroup.com/index.htm), GCC, and [HPE Gray](https://buy.hpe.com/us/en/software/high-performance-computing-software/high-performance-computing-software/hpe-cray-programming-environment/p/1012707351) (only for FORTRAN) compilers
     - Now PGI is part of Nvidia, and it is available through [Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk)
* Compute constructs:
     - parallel and kernel
* Loop constructs:
     - loop, collapse, gang, worker, vector, etc.
* Data management clauses:
     - copy, create, copyin, copyout, delete and present
* Others:
     - reduction, atomic, cache, etc.
* More information about the OpenACC directives can be found in [here](https://www.openacc.org/sites/default/files/inline-files/OpenACC_2_0_specification.pdf)


#### Basic programming structure

~~~c
// C/C++
#include "openacc.h"
#pragma acc <directive> [clauses [[,] clause] . . .] new-line
<code>
~~~

~~~fortran
!! Fortran
use openacc
!$acc <directive> [clauses [[,] clause] . . .]
<code>
~~~


# Compute and loop constructs in OpenACC
#### _kernels_ in C/C++

~~~ c
// Hello_World.c                    | // Hello_World_OpenACC.c
void Print_Hello_World()            | void Print_Hello_World()
{                                   | {
                                    | #pragma acc kernels
  for(int i = 0; i < 5; i++)        |    for(int i = 0; i < 5; i++)
    {                               |      {
      printf("Hello World!\n");     |        printf("Hello World!\n");
    }                               |      }
}                                   | }
~~~


* compilation: ```nvc -fast -Minfo=all -ta=tesla -acc Hello_World.c```
      - The compiler will already give much info; what do you see?


```
$> nvc -fast -Minfo=all -ta=tesla -acc Hello_World.c
Print_Hello_World:
      6, Loop not parallelized: contains call
main:
     14, Print_Hello_World inlined, size=4 (inline) file Hello_World.c (5)
           6, Loop not vectorized/parallelized: contains call
```
* Now add either _kernels_ or _parallel_ directives to vectorize/parallelize the loop
```
$> nvc -fast -Minfo=all -ta=tesla -acc  Hello_World_OpenACC.c

print_hello_world:
      6, Loop is parallelizable
         Generating Tesla code
          6, !$acc loop gang, vector(32) ! blockidx%x threadidx%x

```
* As we can see above the loop is parallelized!.


#### _kernels_ in FORTRAN

~~~ fortran
!! Hello_World.f90                 | !! Hello_World_OpenACC.f90
subroutine Print_Hello_World()     | subroutine Print_Hello_World()
  integer :: i                     |   integer :: i
                                   |   !$acc kernels
  do i = 1, 5                      |   do i = 1, 5
     print *, "hello world"        |     print *, "hello world"
  end do                           |   end do
                                   |  !$acc end kernels
end subroutine Print_Hello_World   | end subroutine Print_Hello_World
~~~

* Compile the _Hello_World.f90_ and compiler tells us that the loop is not parallelized.
```
$> nvfortran -fast -Minfo=all -ta=tesla -acc Hello_World.f90

print_hello_world:
      5, Loop not vectorized/parallelized: contains call
```
* Now run the _Hello_World_OpenACC.f90_ either using _kernels_ or _parallel_ and we can already notice that loop is vectorized/parallelized.
```
$> nvfortran -fast -Minfo=all -ta=tesla -acc  Hello_World_OpenACC.f90

print_hello_world:
      6, Loop is parallelizable
         Generating Tesla code
          6, !$acc loop gang, vector(32) ! blockidx%x threadidx%x
```
> Note: this above example shows you how to create the parallel region using _parallel_ and _kernels_. It is quite useful when multiple regions need to be parallelized. However, the above example has just one parallel region.

# _loop_ and _data management_ clauses

#### for C/C++
* Here we can consider simple vector addition example for the OpenACC _loop_ directive

![](./vector_add.png)

~~~ c
// Vector_Addition.c                  | // Vector_Addition_OpenACC.c
float * Vector_Addition               | float * Vector_Addition
(float *a, float *b, float *c, int n) | (float *a, float *b, float *c, int n)
{                                     | {
                                      | #pragma acc kernels loop
                                      | copyin(a[:n], b[0:n]) copyout(c[0:n])
  for(int i = 0; i < n; i ++)         |   for(int i = 0; i < n; i ++)
    {                                 |     {
      c[i] = a[i] + b[i];             |       c[i] = a[i] + b[i];
    }                                 |     }
  return c;                           |
}                                     | }
~~~
* The _loop_ will parallelize the _for_ loop plus also accommodate other OpenACC _clauses_, for example here _copyin_ and _copyput_.
* The above example needs two vectors to be copied to GPU and one vector needs to send the value back to CPU.
* _copyin_ will create the memory on the GPU and transfer the data from CPU to GPU.
* _copyout_ will create the memory on the GPU and transfer the data from GPU to CPU.

#### for FORTRAN

~~~ fortran

!! Vector_Addition.f90                       | !! Vector_Addition_OpenACC.f90
module Vector_Addition_Mod                   | module Vector_Addition_Mod
  implicit none                              |   implicit none
contains                                     | contains
  subroutine Vector_Addition(a, b, c, n)     |   subroutine Vector_Addition(a, b, c, n)
    !! Input vectors                         |     !! Input vectors
    real(8), intent(in), dimension(:) :: a   |     real(8), intent(in), dimension(:) :: a
    real(8), intent(in), dimension(:) :: b   |     real(8), intent(in), dimension(:) :: b
    real(8), intent(out), dimension(:) :: c  |     real(8), intent(out), dimension(:) :: c
    integer :: i, n                          |     integer :: i, n
                                             |     !$acc kernels loop copyin(a(1:n), b(1:n))
                                             |      copyout(c(1:n))
    do i = 1, n                              |     do i = 1, n
       c(i) = a(i) + b(i)                    |        c(i) = a(i) + b(i)
    end do                                   |     end do
                                             |     !$acc end kernels
  end subroutine Vector_Addition             |   end subroutine Vector_Addition
end module Vector_Addition_Mod               | end module Vector_Addition_Mod
~~~

* Now compile and run the above code as we did previously.

# _reduction_ clause in dot product

![](./dot_product.png){align=center width=250}

#### for C/C++

~~~ c
// Vector_Addition.c                  | // Vector_Addition_OpenACC.c
float * Vector_Addition               | float * Vector_Addition
(float *a, float *b, float *c, int n) | (float *a, float *b, float *c, int n)
{                                     | { float sum=0;
                                      | #pragma acc kernels loop
                                      | reduction(+:sum) copyin(a[:n], b[0:n]) copyout(c[0:n])
  for(int i = 0; i < n; i ++)         |   for(int i = 0; i < n; i ++)
    {                                 |     {
      c[i] = a[i] + b[i];             |       c[i] = a[i] + b[i];
    }                                 |       sum+=c[i];
  return c;                           |     }
}                                     | }
~~~

#### for FORTRAN

~~~ fortran

!! Vector_Addition.f90                       | !! Vector_Addition_Reducion_OpenACC.f90
module Vector_Addition_Mod                   | module Vector_Addition_Mod
  implicit none                              |   implicit none
contains                                     | contains
  subroutine Vector_Addition(a, b, c, n)     |   subroutine Vector_Addition(a, b, c, n)
    !! Input vectors                         |     !! Input vectors
    real(8), intent(in), dimension(:) :: a   |     real(8), intent(in), dimension(:) :: a, b
    real(8), intent(in), dimension(:) :: b   |     real(8):: sum=0
    real(8), intent(out), dimension(:) :: c  |     real(8), intent(out), dimension(:) :: c
    integer :: i, n                          |     integer :: i, n
                                             |     !$acc kernels loop reduction(+:sum)
                                             |      copyin(a(1:n), b(1:n)) copyout(c(1:n))
    do i = 1, n                              |     do i = 1, n
       c(i) = a(i) + b(i)                    |        c(i) = a(i) + b(i)
    end do                                   |        sum = c(i)
                                             |     end do
  end subroutine Vector_Addition             |     !$acc end kernels
end module Vector_Addition_Mod               |   end subroutine Vector_Addition
                                               end module Vector_Addition_Mod
~~~
* _reduction_ _clause_ is needed when we want to sum the array or any counting inside the parallel region; this will increase the performance and avoid the error in the total sum.
* The above example shows how to use them in C/C++ and FORTRAN languages.

# !To follow up more about CUDA and OpenACC programming, please visit:[MOOC course: GPU programming for scientific computing and beyond](https://www.futurelearn.com/courses/gpu-programming-scientific-computing)!
