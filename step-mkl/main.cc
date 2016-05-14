#include <cmath>
#include <cstdio>
#include <omp.h>
#include <cassert>
#include <mkl.h>

void VerifyResult(const int n, float* LU, int* ipiv, float* refA) {

  // Verifying that pivoting did not kick in
  bool pivoted=false;
  for (int i = 0; i < n; i++)
    if (ipiv[i] != i+1) pivoted=true;
  if (pivoted) {
    printf("ERROR: pivoting was used!\n");
    for (int i = 0; i < n; i++) {
      printf("%3d %3d\n", i+1, ipiv[i]);
    }
    exit(1);
  }

  // Verifying that A=LU
  float A[n*n];
  float L[n*n];
  float U[n*n];
  A[:] = 0.0f;
  L[:] = 0.0f;
  U[:] = 0.0f;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++)
      L[i*n + j] = LU[i*n + j];
    L[i*n+i] = 1.0f;
    for (int j = i; j < n; j++)
      U[i*n + j] = LU[i*n + j];
  }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
	A[i*n + j] += L[i*n + k]*U[k*n + j];

  double deviation1 = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      deviation1 += (refA[i*n+j] - A[i*n+j])*(refA[i*n+j] - A[i*n+j]);
    }
  }
  deviation1 /= (double)(n*n);
  if (isnan(deviation1) || (deviation1 > 1.0e-2)) {
    printf("ERROR: LU is not equal to A (deviation1=%e)!\n", deviation1);
    exit(1);
  }


#ifdef VERBOSE
  printf("\n(L-D)+U:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", LU[i*n+j]);
    printf("\n");
  }

  printf("\nL:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", L[i*n+j]);
    printf("\n");
  }

  printf("\nU:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", U[i*n+j]);
    printf("\n");
  }

  printf("\nLU:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", A[i*n+j]);
    printf("\n");
  }

  printf("\nA:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", refA[i*n+j]);
    printf("\n");
  }

  printf("deviation1=%e\n", deviation1);
#endif

}

int main(const int argc, const char** argv) {

  // Problem size and other parameters
#ifndef LU_N
#define LU_N 128
#endif
  const lapack_int  n=LU_N;
  int nMatrices=10000;
  const double HztoPerf = 1e-9*2.0/3.0*double(n*n*n)*nMatrices;

  const size_t containerSize = sizeof(float)*n*n+64;
  // Align on 2 MB to get a fresh new page
  char* dataA = (char*) _mm_malloc(containerSize*nMatrices, (1<<21));
  float referenceMatrix[n*n];

  // Initialize matrices
#pragma omp parallel for schedule(guided)
  for (int m = 0; m < nMatrices; m++) {
    float* matrix = (float*)(&dataA[m*containerSize]);
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
	  matrix[i*n+j] = (float)(i*n+j);
	  sum += matrix[i*n+j];
        }
        sum -= matrix[i*n+i];
        matrix[i*n+i] = 2.0*sum;
    }
    matrix[(n-1)*n+n] = 0.0f; // Touch just in case
  }
  referenceMatrix[0:n*n] = ((float*)dataA)[0:n*n];
  
  // Perform benchmark
  printf("LU decomposition of %d matrices of size %dx%d on %s...\n\n", 
	 nMatrices, n, n,
#ifndef __MIC__
	 "CPU"
#else
	 "MIC"
#endif
	 );

  double rate = 0, dRate = 0; // Benchmarking data
  const int nTrials = 10;
  const int skipTrials = 3; // First step is warm-up on Xeon Phi coprocessor
  printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GFLOP/s");
  for (int trial = 1; trial <= nTrials; trial++) {

    lapack_int ipiv[n];
    // Reference calculation to verify results
    LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, (float*)&dataA[0], n, &ipiv[0]);
    if (trial == 1) VerifyResult(n, (float*)(&dataA[0]), ipiv, referenceMatrix);

    const double tStart = omp_get_wtime(); // Start timing
#pragma omp parallel for schedule(guided) private(ipiv)
    for (int m = 1; m < nMatrices; m++) {
      float* matrixA = (float*)(&dataA[m*containerSize]);
      int err = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, matrixA, n, &ipiv[0]);
    }
    const double tEnd = omp_get_wtime(); // End timing

    if (trial > skipTrials) { // Collect statistics
      rate  += HztoPerf/(tEnd - tStart); 
      dRate += HztoPerf*HztoPerf/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %8.2f %s\n", 
	   trial, (tEnd-tStart), HztoPerf/(tEnd-tStart), (trial<=skipTrials?"*":""));
    fflush(0);
  }
  rate/=(double)(nTrials-skipTrials); 
  dRate=sqrt(dRate/(double)(nTrials-skipTrials)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.2f +- %.2f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");

  _mm_free(dataA);

}
