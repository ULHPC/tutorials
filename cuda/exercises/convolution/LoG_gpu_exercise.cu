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

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Host Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Device Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 __global__ void conv_img_gpu(float *img, float *kernel, float *imgf, int Nx, int Ny, int kernel_size)
{
 
  	
  
  /*
  	Task 2: Write your code implementing the image convolution  here !!!
  	
  	int tid = threadIdx.x;               
  	int iy = blockIdx.x + (kernel_size - 1)/2;  
  	int ix = threadIdx.x + (kernel_size - 1)/2; 
  	int idx = iy*Nx +ix;                        
  	int K2 = kernel_size*kernel_size; 
  	int center = (kernel_size -1)/2;	
  	
  	....
  */
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


  /* Allocate CPU memory */
  float *img, *imgf, *kernel;
  img = (float*)malloc(Nx*Ny*sizeof(float));
  imgf = (float*)malloc(Nx*Ny*sizeof(float));
  kernel = (float*)malloc(kernel_size*kernel_size*sizeof(float));  
  
  
  load_image(finput, Nx, Ny, img);
  calculate_kernel(kernel_size, sigma, kernel);
  

  /* Allocate GPU memory*/ 
  float *d_img, *d_imgf, *d_kernel;
  cudaMalloc(&d_img,Nx*Ny*sizeof(float));
  cudaMemcpy(d_img, img, Nx*Ny*sizeof(float),cudaMemcpyHostToDevice);
  
  /*
  	Task 3: Write your code here! (2 lines)
  	
 */
  

 
 /* Task 4a: Complete the statements!
 
  Nblocks = ;
  Nthreads = ;
  
  */
  
  cudaEventRecord(start);
  
  /* Task 4b: Complete the statement to launch your kernel
  
  conv_img_gpu<<<......>>>(......);
  
  */
  
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
