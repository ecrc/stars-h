#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "starsh.h"

void starsh_kernel_drsdd(int nrows, int ncols, double *D, double *U, double *V,
        int *rank, int maxrank, int oversample, double tol, double *work,
        int lwork, int *iwork)
{
    int mn = nrows < ncols ? nrows : ncols;
    int mn2 = maxrank+oversample;
    if(mn2 > mn)
        mn2 = mn;
    //size_t svdqr_lwork = (4*mn2+7)*mn2;
    //if(svdqr_lwork < ncols)
    //    svdqr_lwork = ncols;
    double *X, *Q, *tau, *svd_U, *svd_S, *svd_V, *svdqr_work;
    X = work;
    Q = X+(size_t)ncols*mn2;
    svd_U = Q+(size_t)nrows*mn2;
    svd_S = svd_U+(size_t)mn2*mn2;
    tau = svd_S;
    svd_V = svd_S+mn2;
    svdqr_work = svd_V+(size_t)ncols*mn2;
    int svdqr_lwork = lwork-mn2*(2*ncols+nrows+mn2+1);
    int iseed[4] = {0, 0, 0, 1};
    // Generate random matrix X
    LAPACKE_dlarnv_work(3, iseed, nrows*mn2, X);
    // Multiply by random matrix
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, mn2,
            ncols, 1.0, D, nrows, X, ncols, 0.0, Q, nrows);
    // Get Q factor of QR factorization
    LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, nrows, mn2, Q, nrows, tau,
            svdqr_work, svdqr_lwork);
    LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, nrows, mn2, mn2, Q, nrows, tau,
            svdqr_work, svdqr_lwork);
    // Multiply Q by initial matrix
    cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, mn2, ncols,
            nrows, 1.0, Q, nrows, D, nrows, 0.0, X, mn2);
    // Get SVD of result to reduce rank
    LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', mn2, ncols, X, mn2, svd_S,
            svd_U, mn2, svd_V, mn2, svdqr_work, svdqr_lwork, iwork);
    // Get rank, corresponding to given error tolerance
    *rank = starsh__dsvfr(mn2, svd_S, tol);
    if(*rank < mn/2 && *rank <= maxrank)
    // If far-field block is low-rank
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, *rank,
                mn2, 1.0, Q, nrows, svd_U, mn2, 0.0, U, nrows);
        for(size_t i = 0; i < *rank; i++)
        {
            cblas_dcopy(ncols, svd_V+i, mn2, V+i*ncols, 1);
            cblas_dscal(ncols, svd_S[i], V+i*ncols, 1);
        }
    }
    else
    // If far-field block is dense, although it was initially assumed
    // to be low-rank. Let denote such a block as false far-field block
        *rank = -1;
}
