#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "starsh.h"

void starsh_kernel_drsdd(int nrows, int ncols, double *D, double *U, double *V,
        int *rank, int maxrank, int oversample, double tol, double *work,
        int lwork, int *iwork)
//! 1-way randomized SVD approximation of a dense double precision matrix.
/* @param[in] nrows: Number of rows of a matrix.
 * @param[in] ncols: Number of columns of a matrix.
 * @param[in,out] D: Pointer to dense matrix.
 * @param[out] U: Pointer to low-rank factor `U`.
 * @param[out] V: Pointer to low-rank factor `V`.
 * @param[out] rank: Address of rank variable.
 * @param[in] maxrank: Maximum possible rank.
 * @param[in] oversample: Rank oversampling.
 * @param[in] tol: Relative error for approximation.
 * @param[in] work: Working array.
 * @param[in] lwork: Size of `work` array.
 * @param[in] iwork: Temporary integer array.
 *
 * Uses 1-way randomized SVD algorithm. Works only with fast decay of
 * singular values.
 * */
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
    int info = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', mn2, ncols, X, mn2,
            svd_S, svd_U, mn2, svd_V, mn2, svdqr_work, svdqr_lwork, iwork);
    if(info != 0)
        STARSH_WARNING("LAPACKE_dgesdd_work info=%d", info);
    // Get rank, corresponding to given error tolerance
    *rank = starsh__dsvfr(mn2, svd_S, tol);
    if(info == 0 && *rank < mn/2 && *rank <= maxrank)
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
