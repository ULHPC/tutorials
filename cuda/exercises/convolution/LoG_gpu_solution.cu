/*
 Copyright (c) Loizos Koutsantonis <loizos.koutsantonis@uni.lu>

 Description : CUDA code implementing convolution of an image with a 
 LoG kernel. 
 Implemented for educational purposes.

 This program is free software: you can redistribute it and/or modify
 it under the terms of the NVIDIA Software License Agreement and CUDA 
 Supplement to Software License Agreement. 

 University of Luxembourg - HPC 
 November 2020
*/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>

#define pi 3.14159265359




/*
Function load_image:
Load BW Image from dat (Ascii) file (Host Function)
Nx,Ny: Image Dimensions
fname: fielename (char)
img: float vector containing the image pixels
*/
void load_image(char *fname, int Nx, int Ny, float  *img)
{
  FILE *fp;
  
  fp=fopen(fname,"r");
  
  for (int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++)
      fscanf(fp,"%f ",&img[i*Nx+j]);
     fscanf(fp,"\n");
  }
  
  fclose(fp);
}





/*
Function save_image:
Save BW Image to dat (Ascii) file (Host Function)
Nx,Ny: Image Dimensions
fname: fielename (char)
img: float vector containing the image pixels
*/
void save_image(char *fname, int Nx, int Ny, float  *img)
{
  FILE *fp;
  
  fp=fopen(fname,"w");
  
  for (int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++)
      fprintf(fp,"%10.3f ",img[i*Nx+j]);
     fprintf(fp,"\n");
  }
  
  fclose(fp);
}




/*
Function calculate_kernel:
Calculate filter coefficients of LoG filter
and save them to a vector (Host Function)
kernel_size: Length of filter window in pixels (same for x and y)
sigma: sigma of the Gaussian kernel (float) given in pixels
kernel: float vector hosting the kernel coefficients
*/
void calculate_kernel(int kernel_size, float sigma, float *kernel){

  int Nk2 = kernel_size*kernel_size;
  float x,y, center;

  center = (kernel_size-1)/2.0;
  
  for (int i = 0; i<Nk2; i++){
    x = (float)(i%kernel_size)-center;
    y =(float)(i/kernel_size)-center;
    kernel[i] = -(1.0/pi*pow(sigma,4))*(1.0 - 0.5*(x*x+y*y)/(sigma*sigma))*exp(-0.5*(x*x+y*y)/(sigma*sigma));
  }

}





/*
Function conv_img_cpu:
Convolve image with the specified kernel  (Host Function)
img: float vector containing the original image pixels
kernel: float vector hosting the kernel coefficients
imgf: float vector containing the result of the convolution
Nx,Ny: Original Image Dimensions
kernel_size: Length of filter window in pixels (same for x and y)
*/
void conv_img_cpu(float *img, float *kernel, float *imgf, int Nx, int Ny, int kernel_size)
{
  
  float sum = 0;
  int center = (kernel_size -1)/2;;
  int ii, jj;
  
  for (int i = center; i<(Ny-center); i++)
    for (int j = center; j<(Nx-center); j++){
      sum = 0;
      for (int ki = 0; ki<kernel_size; ki++)
	for (int kj = 0; kj<kernel_size; kj++){
	  ii = kj + j - center;
	  jj = ki + i - center;
	  sum+=img[jj*Nx+ii]*kernel[ki*kernel_size + kj];
	}
      imgf[i*Nx +j] = sum;
    }
}





/*
Function conv_img_cpu:
Convolve image with the specified kernel  (Device Function)
img: float vector containing the original image pixels
kernel: float vector hosting the kernel coefficients
imgf: float vector containing the result of the convolution
Nx,Ny: Original Image Dimensions
kernel_size: Length of filter window in pixels (same for x and y)
*/
 __global__ void conv_img_gpu(float *img, float *kernel, float *imgf, int Nx, int Ny, int kernel_size)
{
  
  //local ID of each thread (withing block) 
  int tid = threadIdx.x;    
  
  //each block is assigned to a row of an image, iy index of y value                  
  int iy = blockIdx.x + (kernel_size - 1)/2;  
  
  //each thread is assigned to a pixel of a row, ix index of x value
  int ix = threadIdx.x + (kernel_size - 1)/2; 
  
  //idx global index (all blocks) of the image pixel 
  int idx = iy*Nx +ix;                        
 
 
 //total number of kernel elements
  int K2 = kernel_size*kernel_size; 
  
  //center of kernel in both dimensions          
  int center = (kernel_size -1)/2;		
  
  //Auxiliary variables
  int ii, jj;
  float sum = 0.0;

 
 /*
 Define a vector (float) sdata[] that will be hosted in shared memory,
 *extern* dynamic allocation of shared memory: kernel<<<blocks,threads,memory size to be allocated in shared memory>>>
*/  
  extern __shared__ float sdata[];         


/*Transfer data frm GPU memory to shared memory 
tid: local index, each block has access to its local shared memory
e.g. 100 blocks -> 100 allocations/memory spaces
Eeah block has access to the kernel coefficients which are store in shared memory
Important: tid index must not exceed the size of the kernel*/

  if (tid<K2)
    sdata[tid] = kernel[tid];             
  
  
  
  
  //Important. Syncronize threads before performing the convolution.
  //Ensure that shared memory is filled by the tid threads
  
  __syncthreads();			  
  						
  
  
  
  /*
  Convlution of image with the kernel
  Each thread computes the resulting pixel value 
  from the convolution of the original image with the kernel;
  number of computations per thread = size_kernel^2
  The result is stored to imgf
  */
  
  if (idx<Nx*Ny){
    for (int ki = 0; ki<kernel_size; ki++)
      for (int kj = 0; kj<kernel_size; kj++){
	ii = kj + ix - center;
	jj = ki + iy - center;
	sum+=img[jj*Nx+ii]*sdata[ki*kernel_size + kj];
      }
  
    imgf[idx] = sum;
  }
  
 
}



int main(int argc, char *argv[]){
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
float milliseconds = 0;
  int Nx, Ny;
  int kernel_size;
  float sigma;
  char finput[256], foutput[256];
  int Nblocks, Nthreads;


  
  sprintf(finput,"lux_bw.dat");
  sprintf(foutput,"lux_output.dat") ;


  
  Nx = 600;
  Ny = 570;

  kernel_size = 5;
  sigma = 0.8;



  
  /* Allocate CPU memory 
     Vector Representation of Images and Kernel 
     (Original Image, Kernel, Convoluted Image) */
  float *img, *imgf, *kernel;
  
  img = (float*)malloc(Nx*Ny*sizeof(float));
  imgf = (float*)malloc(Nx*Ny*sizeof(float));
  kernel = (float*)malloc(kernel_size*kernel_size*sizeof(float));  
  
  

  /* Allocate GPU memory 
     Vector Representation of Images and Kernel 
     (Original Image, Kernel, Convoluted Image) */
  
  float *d_img, *d_imgf, *d_kernel;
  
  cudaMalloc(&d_img,Nx*Ny*sizeof(float));
  cudaMalloc(&d_imgf,Nx*Ny*sizeof(float));
  cudaMalloc(&d_kernel,kernel_size*kernel_size*sizeof(float));
  
  load_image(finput, Nx, Ny, img);
  calculate_kernel(kernel_size, sigma, kernel);

  cudaMemcpy(d_img, img, Nx*Ny*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel,kernel, kernel_size*kernel_size*sizeof(float),cudaMemcpyHostToDevice);

  Nblocks = Ny - (kernel_size-1);
  Nthreads = Nx - (kernel_size-1);
  
  //conv_img_cpu(img, kernel, imgf, Nx, Ny, kernel_size);
  cudaEventRecord(start);
  conv_img_gpu<<<Nblocks, Nthreads, kernel_size*kernel_size*sizeof(float)>>>(d_img, d_kernel, d_imgf, Nx, Ny, kernel_size);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  cudaMemcpy(imgf, d_imgf, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
  save_image(foutput, Nx, Ny, imgf);
  
  printf("\n");
  printf("Convolution Completed !!! \n");
  printf("Ellapsed Time (GPU): %16.10f ms\n", milliseconds);
  printf("\n");
  
  
  
  free(img);
  free(imgf);
  free(kernel);

  cudaFree(d_img);
  cudaFree(d_imgf);
  cudaFree(d_kernel);
}
