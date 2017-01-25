#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "stars.h"
#include "misc.h"

void starsh_kernel_drsdd(int nrows, int ncols, double *D, Array *U, Array *V,
        int *rank, int maxrank, int oversample, double tol, double *work,
        int lwork, int *iwork)
{
    int mn = nrows < ncols ? nrows : ncols;
    int mn2 = maxrank+oversample;
    if(mn2 > mn)
        mn2 = mn;
    size_t svd_lwork = (4*(size_t)mn2+7)*mn2;
    if(svd_lwork < ncols)
        svd_lwork = ncols;
    double *X, *Q, *tau, *svd_U, *svd_S, *svd_V, *svd_work, *out_U, *out_V;
    X = work;
    Q = X+(size_t)ncols*mn2;
    svd_U = Q+(size_t)nrows*mn2;
    svd_S = svd_U+(size_t)mn2*mn2;
    tau = svd_S;
    svd_V = svd_S+mn2;
    svd_work = svd_V+(size_t)ncols*mn2;
    out_U = U->data;
    out_V = V->data;
    // Generate random matrix X
    for(size_t i = 0; i < mn2; i++)
        for(size_t j = 0; j < ncols; j++)
            X[i*ncols+j] = randn();
    // Multiply by random matrix
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, mn2,
            ncols, 1.0, D, nrows, X, ncols, 0.0, Q, nrows);
    // Get Q factor of QR factorization
    LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, nrows, mn2, Q, nrows, tau, svd_work,
            svd_lwork);
    LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, nrows, mn2, mn2, Q, nrows, tau,
            svd_work, svd_lwork);
    // Multiply Q by initial matrix
    cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, mn2, ncols,
            nrows, 1.0, Q, nrows, D, nrows, 0.0, X, mn2);
    // Get SVD of result to reduce rank
    LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', mn2, ncols, X, mn2, svd_S,
            svd_U, mn2, svd_V, mn2, svd_work, svd_lwork, iwork);
    // Get rank, corresponding to given error tolerance
    *rank = starsh__dsvfr(mn2, svd_S, tol);
    if(*rank < mn/2 && *rank <= maxrank)
    // If far-field block is low-rank
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, *rank,
                mn2, 1.0, Q, nrows, svd_U, mn2, 0.0, out_U, nrows);
        for(size_t i = 0; i < *rank; i++)
        {
            cblas_dcopy(ncols, svd_V+i, mn2, out_V+i*ncols, 1);
            cblas_dscal(ncols, svd_S[i], out_V+i*ncols, 1);
        }
        U->shape[1] = *rank;
        V->shape[1] = *rank;
    }
    else
    {
        *rank = -1;
    }
}
