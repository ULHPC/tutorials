#include <stdio.h>
#include <openacc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


#define pi 3.14159265359
#define f 0.02

void initialize(float *F, float *Fnew, int n, int m)
{
  memset(F,    0, n*m*sizeof(float));
  memset(Fnew, 0, n*m*sizeof(float));
  
  for (int j = 0; j < m; j++){
    F[j] = 1.0*cos(2*pi*j*f);
    Fnew[j] = F[j];
    F[(n-1)*m +j] = 1.0*cos(2*pi*j*f);
    Fnew[(n-1)*m +j] =  F[(n-1)*m +j];
  }
  
  for (int j = 0; j < n; j++){
    F[j*m] = 1.0;
    Fnew[j*m] = F[j*m];
    F[(j+1)*m-1] = 1.0;
    Fnew[(j+1)*m-1] = F[(j+1)*m-1];
  }
  
}


float calcTempStep(float *restrict F, float *restrict Fnew, int n, int m)
{
  float Fu, Fd, Fl, Fr;
  float error = 0.0;
  
  //Code here!
  for (int i = 1; i < n-1; i++){
    //Code Here!
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




void update(float *restrict F, float  *restrict Fnew, int n, int m)
{
  
  //Code Here!
  for (int i = 0; i < n; i++)
    //Code Here!
    for (int j = 0; j < m; j++ )
      F[i*m+j] = Fnew[i*m+j]; 
  
  
}

void deallocate(float *F, float *Fnew)
{
  free(F);
  free(Fnew);
}






void printDist(float *F, int n, int m, char *fname)
{
  FILE *fa;
  fa=fopen(fname,"w");
  for (int i = 0; i < n; i++){
    for (int j = 0; j < m; j++ )
      fprintf(fa, "%8.3f ", F[i*m+j]); 
    fprintf(fa, "\n");
  }
  fclose(fa); 
  
  
}



int main(int argc, char** argv)
{
  const int n = 1024;
  const int m = 1024;
  const int miter = 10000;
  const float thres = 1.0e-6;
  float error = 1.0;
  float *restrict T    = (float*)malloc(n*m*sizeof(float));
  float *restrict Tnew = (float*)malloc(n*m*sizeof(float));
  struct timeval start_time, stop_time, elapsed_time;
  
  
  
  
  
  
  initialize(T, Tnew, n, m);
  
  printDist(T, n, m, "temp.in");
  
  gettimeofday(&start_time,NULL);
  
  int iter = 0;
  
  
  //Code Here!
  while ((iter < miter )&& (error > thres))
    {
      error = calcTempStep(T, Tnew, n, m);
      
      update(T, Tnew, n, m);
      
      if(iter % 50 == 0) printf("Iterations = %5d, Error = %16.10f\n", iter, error);
      
      iter++;
    }
  
  gettimeofday(&stop_time,NULL);
  timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine
  printf("\nTotal Iterations = %16d \nError            = %16.10f \nTotal time (sec) = %16.3f \n", iter, error, elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
  
  //printDist(T, n, m, "temp.out");
  
  deallocate(T, Tnew);
  
  return 0;
}
