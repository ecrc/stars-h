#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "lapacke.h"

double randn()
{
    // Random Gaussian generation of double, got it from the Internet
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if(phase == 0)
    {
        do
        {
            double U1 = (double)rand()/RAND_MAX;
            double U2 = (double)rand()/RAND_MAX;
            V1 = 2*U1-1;
            V2 = 2*U2-1;
            S = V1*V1+V2*V2;
        } while(S >= 1 || S == 0);
        X = V1*sqrt(-2*log(S)/S);
    }
    else
    {
        X = V2*sqrt(-2*log(S)/S);
    }
    phase = 1-phase;
    return X;
}

void dmatrix_print(int m, int n, double *A)
{
    // Print matrix
    int i, j;
    // Row cycle
    for(i = 0; i < m; i++)
    {
        // For each element of a row
        for(j = 0; j < n; j++)
            printf("%f ", A[j*m+i]);
        printf("\n");
    }
}

void dmatrix_randn(int m, int n, double *A)
{
    // Generate a random matrix with normal(0,1) distribution
    // Parameters:
    //   n: number of rows,
    //   m: number of columns,
    //   A: pointer to the matrix
    int i, size = n*m;
    for(i = 0; i < size; i++)
    {
        A[i] = randn();
    }
}

void dmatrix_qr(int m, int n, double *A, double *Q, double *R)
{
    // Computes QR of a matrix
    int mn = n > m ? m : n;
    int i, j;
    double tau[mn];
    double zero = 0.0;
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, m, tau);
    if(R != NULL)
    {
        // Column cycles
        for(i = 0; i < mn; i++)
        {
            // For each element in a column
            for(j = 0; j <= i; j++)
                R[i*mn+j] = A[i*m+j];
            for(j = i+1; j < mn; j++)
                R[i*mn+j] = zero;
        }
        for(i = mn; i < n; i++)
            for(j = 0; j < mn; j++)
                R[i*mn+j] = A[i*m+j];
    }
    if(Q != NULL)
    {
        int size = m*mn;
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, mn, A, m, tau);
        if(Q != A)
        {
            for(i = 0; i < size; i++)
            {
                Q[i] = A[i];
            }
        }
    }
    return;
}

void dmatrix_lr(int m, int n, double *A, double tol, int *rank, double **U,
        double **S, double **V)
{
    // Computes SVD of a matrix and computes rank for approximation with
    // given relative error in Frobenius norm
    int mn = m > n ? n : m;
    int i;
    double stol;
    double *UU = (double *)malloc(m*mn*sizeof(double));
    double *SS = (double *)malloc(mn*sizeof(double));
    double *SS2 = (double *)malloc(mn*sizeof(double));
    double *VV = (double *)malloc(mn*n*sizeof(double));
    LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S', m, n, A, m, SS, UU, m, VV, mn);
    *rank = mn;
    SS2[mn-1] = SS[mn-1]*SS[mn-1];
    for(i = mn-2; i >= 0; i--)
        SS2[i] = SS[i]*SS[i]+SS2[i+1];
    stol = tol*tol*SS2[0];
    for(i = 0; i < mn; i++)
        if(SS2[i] < stol)
        {
            *rank = i;
            break;
        }
    free(SS2);
    *U = UU;
    *S = SS;
    *V = VV;
    return;
}
