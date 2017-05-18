/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * @file backends/sequential/kernels/drsdd2.c
 * @version 1.0.0.2
 * @author Aleksandr Mikhalev
 * @date 16 May 2017
 * */

#include "common.h"
#include "starsh.h"

void starsh_kernel_drsdd2(int nrows, int ncols, double *D, double *U, double *V,
        int *rank, int maxrank, int oversample, double tol, double *work,
        int lwork, int *iwork)
//! 2-way randomized SVD approximation of a dense double precision matrix.
/*! @ingroup approximations
 * @param[in] nrows: Number of rows of a matrix.
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
 * Uses 2-way randomized SVD algorithm. Works only with fast decay of
 * singular values. Less accurate, than 1-way randomized SVD.
 * */
{
    int mn = nrows < ncols ? nrows : ncols;
    int mn2 = maxrank+oversample;
    if(mn2 > mn)
        mn2 = mn;
    size_t svdqr_lwork = (4*(size_t)mn2+7)*mn2;
    if(svdqr_lwork < ncols)
        svdqr_lwork = ncols;
    if(svdqr_lwork < nrows)
        svdqr_lwork = nrows;
    double *X, *Y, *QX, *QY, *tau, *R, *svd_U, *svd_S, *svd_V, *svdqr_work;
    X = work;
    Y = X+(size_t)ncols*mn2;
    QX = Y+(size_t)nrows*mn2;
    QY = QX+(size_t)nrows*mn2;
    R = D;
    svd_U = X;
    svd_S = QY+(size_t)ncols*mn2;
    tau = svd_S;
    svd_V = Y;
    svdqr_work = svd_S+mn2;
    int iseed[4] = {0, 0, 0, 1};
    // Generate random matrices X and Y
    LAPACKE_dlarnv_work(3, iseed, ncols*mn2, X);
    LAPACKE_dlarnv_work(3, iseed, nrows*mn2, Y);
    // Multiply by random matrices
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, mn2,
            ncols, 1.0, D, nrows, X, ncols, 0.0, QX, nrows);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncols, mn2,
            nrows, 1.0, D, nrows, Y, nrows, 0.0, QY, ncols);
    // Get Q factor of QR factorizations and R factor for one of matrices
    LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, nrows, mn2, QX, nrows, tau,
            svdqr_work, svdqr_lwork);
    LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, nrows, mn2, mn2, QX, nrows, tau,
            svdqr_work, svdqr_lwork);
    LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, ncols, mn2, QY, ncols, tau,
            svdqr_work, svdqr_lwork);
    for(size_t i = 0; i < mn2; i++)
    {
        cblas_dcopy(i+1, QY+i*ncols, 1, R+i, mn2);
        for(size_t j = i+1; j < mn2; j++)
            R[i+j*mn2] = 0.;
    }
    LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, ncols, mn2, mn2, QY, ncols, tau,
            svdqr_work, svdqr_lwork);
    // Multiply Q by random matrix
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mn2, mn2, nrows,
            1.0, Y, nrows, QX, nrows, 0.0, X, mn2);
    // Solve system
    LAPACKE_dgesv_work(LAPACK_COL_MAJOR, mn2, mn2, X, mn2, iwork, R, mn2);
    // Get SVD of result to reduce rank
    LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', mn2, mn2, R, mn2, svd_S,
            svd_U, mn2, svd_V, mn2, svdqr_work, svdqr_lwork, iwork);
    // Get rank, corresponding to given error tolerance
    *rank = starsh__dsvfr(mn2, svd_S, tol);
    if(*rank < mn/2 && *rank <= maxrank)
    // If far-field block is low-rank
    {
        for(size_t i = 0; i < *rank; i++)
            cblas_dscal(mn2, svd_S[i], svd_U+i*mn2, 1);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, *rank,
                mn2, 1.0, QX, nrows, svd_U, mn2, 0.0, U, nrows);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, ncols, *rank,
                mn2, 1.0, QY, ncols, svd_V, mn2, 0.0, V, ncols);
    }
    else
    // If far-field block is dense, although it was initially assumed
    // to be low-rank. Let denote such a block as false far-field block
        *rank = -1;
}
