#include "lapacke.h"
#include "cblas.h"


void reconstructor(double *Cmm,double *Ctm, long nx, long ny,long lda,long ldb);
void compute_Cee_Cvv(int nMeasTS, int nMeas,int nact, double *Cmm, double *Cpp, double *Cpm, double *R, double *Dx, double *Cee, double *Cvv);

//void _dtrsm(int M, int N, double *A, int lda, double *B, int ldB);
//void _dsyr2k(int N, int M, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);
//void _dsymm(int M, int N, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);
//void _dlacpy(int M,int N, double *A, int lda, double *B, int ldb);
//void _dgemm(int M,int N,int K,double alpha, double *A, int lda, double *B,int ldb,double beta, double *C,int ldc);


void add(char Trans,int M,int N,double alpha,double *A,double beta,double *B);
void scaldiag(int M,double alpha, double *A);
void restore_symm(char uplo,int M, double *A,int lda);
