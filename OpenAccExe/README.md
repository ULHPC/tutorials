[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/cuda/exercises/convolution/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/cuda/exercises/convolution/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/cuda/exercises/convolution/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Solving the Laplace Equation on GPU with OpenAcc

     Copyright (c) 2020-2021 L. Koutsantonis, UL HPC Team <hpc-team@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/cuda/exercises/convolution/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/cuda/exercises/convolution/slides.pdf)



<br />
<br />

--------------------
## Pre-requisites 

Ensure you are able to [connect to the UL HPC cluster](https://hpc.uni.lu/users/docs/access.html).
In particular, recall that the `module` command **is not** available on the access frontends.


Access to ULHPC cluster  (here iris): 

```bash
(laptop)$> ssh iris-cluster
```

/!\ Advanced (but recommended) best-practice:
Always work within an GNU Screen session named with 'screen -S <topic>' (Adapt accordingly)
IF not yet done, copy ULHPC .screenrc in your home:

```bash
(access)$> cp /etc/dotfiles.d/screen/.screenrc ~/
```

Now you'll need to pull the latest changes in your working copy of the [ULHPC/tutorials](https://github.com/ULHPC/tutorials) you should have cloned in `~/git/github.com/ULHPC/tutorials` (see ["preliminaries" tutorial](../../preliminaries/))

``` bash
(access)$> cd ~/git/github.com/ULHPC/tutorials
(access)$> git pull

```

## Objectives

The objective of this tutorial is to show how the OpenAcc directives can be used to accelerate a numerical solver commonly used in engineering and scientific applications. After completing the exercise of this tutorial you would be able to: 

* Transfer data from host to  device using the data directives,

* Accelerate a nested loop application with the loop directives, and,

* Use the reduction clause to perform summation on variables or  elements of a vector.


## The Laplace Equation

* The Laplace differential equation in 2D is given by:

$$ \nabla^2 F = \frac{d^2F}{dx^2} + \frac{d^2F}{dy^2} = 0 $$

* It models a distribution at steady state or equilibrium in a 2D space (e.g. Temperature Distribution). 

* The Laplace differential equation can be solved using the Jacobi method if the boundary conditions are known (e.g. the temperature at the edges of the physical region of interest)

An example of a 2D problem is demonstrated in the figures bellow. The first figure presents the temperature at the edges of a plane. The solution of the Laplacian equation providing  the steady state temperature distribution was calculated for the given boundary condition using the Jacobi method and is shown in the second figure.

<p>
    <img src="images/tinp.png" alt  width="650;">
    <center><em>Boundary Conditions</em></center>
</p>


<br />
<br />



<p>
    <img src="images/tout.png" alt  width="650;">
    <center><em>Solution</em></center>
</p>



<br />
<br />



## The Jacobi method

* Iterative method for solving a system of equations:

$$ Ax = b $$

where, the elements $A$ and $b$  are constants and $x$ is the vector with the unknowns.

* At each iteration, the elements $x$ are updated using their previous estimations by:

$$ x_i = \frac{1}{A_{ii}}(b_i - \sum_{i \ne j} A_{ij}x^{k-1}_j) $$

* An error metric is calculated at each iteration $k$ over the elements $x_i$:

$$ Error = \sum_i (x^k_i - x^{(k-1)}_i)^2 $$

* The algorithm terminates when this error becomes smaller than a predefined threshold:

$$ Error<Threshold $$


## Solving the Laplace Equation using the Jacobi method

* Second order derivatives can be calculated numerically using a small enough value of $\delta$:

$$\frac{d^2f}{dx^2} = \frac{1}{\delta ^2}(f(x+\delta, y) - 2f(x,y)+f(x-\delta,y))$$

$$\frac{d^2f}{dy^2} = \frac{1}{\delta ^2}(f(x, y +\delta) - 2f(x,y)+f(x, y-\delta))$$

* Substituting the numerical second order derivatives in Laplace equation gives:

$$f(x,y) = \frac{1}{4}(f(x, y +\delta) + f(x, y -\delta) + f(x+\delta, y) + f(x-\delta, y)) $$

The above equation results to a stencil of four points shown in the figure bellow.

<p>
    <center><img src="images/stencil.png" alt  width="400;"></center>
    <center><em>Stencil of 4 points</em></center>
</p>


## Implementation of Jacobi Method in C

### CPU Implementation
A serial code implementing the Jacobi method employs a nested loop to compute the elements of a matrix at each iteration. At each iterattion, the error (distance metric) is calculated over these elements. This calculated error is monitored to terminate the iterative Jacobi algorithm: 

```c
while ((iter < miter )&& (error > thres))
    {
      error = calcTempStep(T, Tnew, n, m);
      
      update(T, Tnew, n, m);
      
      if(iter % 50 == 0) printf("Iterations = %5d, Error = %16.10f\n", iter, error);
      
      iter++;
    }
```


```c
float calcTempStep(float *restrict F, float *restrict Fnew, int n, int m)
{
  float Fu, Fd, Fl, Fr;
  float error = 0.0;
  
 
  for (int i = 1; i < n-1; i++){
    for (int j = 1; j < m-1; j++){
      Fu = F[(i-1)*m + j];
      Fd = F[(i+1)*m + j];
      Fl = F[i*m + j - 1];
      Fr = F[i*m + j + 1];
      Fnew[i*m+j] = 0.25*(Fu + Fd + Fl + Fr);
      error += (Fnew[i*m+j] - F[i*m+j])*(Fnew[i*m+j] - F[i*m+j]);
    }
  }
  
  
  return error;
}
```



```c
void update(float *restrict F, float  *restrict Fnew, int n, int m)
{
  
  for (int i = 0; i < n; i++)
    //Code Here!
    for (int j = 0; j < m; j++ )
      F[i*m+j] = Fnew[i*m+j]; 
  
  
}
```

<br />
<br />

## Exercise: Parallelize the Jacobi iteration with OpenAcc

Follow the steps bellow to accelerate the Jacobi solver using the OpenAcc directives. 


***Task 1:*** If you do not have yet the UL HPC tutorial repository, clone it. Update to the latest version.

```bash
ssh iris-cluster
mkdir -p ~/git/github.com/ULHPC
cd  ~/git/github.com/ULHPC
git clone https://github.com/ULHPC/tutorials.git
cd tutorials/OpenAccExe/exercise/
git stash && git pull -r && git stash pop
```
<br />




***Task 2:*** Get an interactive GPU job

```bash
### ... either directly - dedicate 1/4 of available cores to the management of GPU card
$> si-gpu -c7
# /!\ warning: append -G 1 to really reserve a GPU
# salloc -p interactive --qos debug -C gpu -c7 -G 1 --mem-per-cpu 27000

### ... or using the HPC School reservation 'hpcschool-gpu'
salloc --reservation=hpcschool-gpu -p interactive -C gpu --ntasks-per-node 1 -c7 -G 1
<br />


***Task 3:*** Load the required modules

```bash
module load compiler/PGI/19.10-GCC-8.3.0-2.32
module load compiler/GCC
```
<br />


***Task 4:*** Following the steps `1` to `3` provided bellow write a CUDA kernel for the computation of the convolution operator.

Open the source file `LoG_gpu_exercise.cu` with your favorite editor (e.g. `emacs LoG_gpu_exercise.cu`). The CUDA kernel is already defined:

```c
void conv_img_cpu(float *img, float *kernel, float *imgf, int Nx, int Ny, int kernel_size)
```
where `*img` is a pointer to the original image vector, `*kernel` is a pointer to the convolution kernel vector, `*imgf` is a pointer to the convoluted image, `Nx` and `Ny` are the dimensions of both the original and convoluted image, and `kernel_size` is the dimension of the convolution kernel.


```



### Compile and run your code

 The NVIDIA CUDA compiler 'nvcc' is used to compile the source code containing both the host and device functions.
The non CUDA part of the code will be forwarded to a general purpose host compiler (e.g. `gcc`).
As you have seen, the GPU functionsare declared using some annotations (e.g. `__global__,` `__device__`) distinguishing them from the host code.

In simple words, you can compile your code using:

```bash
nvcc -arch=compute_70 -o ./$exe $src
```
where `nvcc` is the keyword for the nvcc compiler,
`$src` is the name of the source file to compile ( e.g.`LoG_gpu.cu` is passed as the source file to compile,
the `o` flag is used to specify the name `$exe` of the compiled program, and
the `arch` points to the GPU architecture for which the source file must be compiled. `sm_70` indicates the Volta GPU architecture. Use `-lm` to link the math library to your executable.

To run your executable file interactively, just use:
```bash
./$exe $arg1 $arg2 ... $argn
```
where `$arg1`, `$arg2`, `...`, `$argn` are the appropriate arguments (if any).

<br />
<br />



### Experimentation with convolution parameters

Try to change the parameters `sigma` and `kernel_size` in your main function. Try to use a large enough convolution kernel size. Compile the modified source code and use `nvprof` to profile your application.
Do you observe any difference in the execution time of the GPU kernel?

Try to implement the GPU kernel without using the shared memory. In this case the CUDA kernel is implemented as follows:

```c

if (idx<Nx*Ny){
    for (int ki = 0; ki<kernel_size; ki++)
        for (int kj = 0; kj<kernel_size; kj++){
	       ii = kj + ix - center;
	       jj = ki + iy - center;
	sum+=img[jj*Nx+ii]*kernel[ki*kernel_size + kj];
    }
    imgf[idx] = sum;
  }
```

Again, recompile your source file and profile your application.
Can you observe any difference in the execution time of the GPU kernel?

<br />

