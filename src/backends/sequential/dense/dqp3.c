/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/sequential/dense/dqp3.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-13
 * */

#include "common.h"
#include "starsh.h"

void starsh_dense_dlrqp3(int nrows, int ncols, double *D, double *U,
        double *V, int *rank, int maxrank, double tol, double *work,
        int lwork, int *iwork)
//! Rank-revealing QR approximation of a dense double precision matrix.
/*! @ingroup approximations
 * @param[in] nrows: Number of rows of a matrix.
 * @param[in] ncols: Number of columns of a matrix.
 * @param[in,out] D: Pointer to dense matrix.
 * @param[out] U: Pointer to low-rank factor `U`.
 * @param[out] V: Pointer to low-rank factor `V`.
 * @param[out] rank: Address of rank variable.
 * @param[in] maxrank: Maximum possible rank.
 * @param[in] tol: Relative error for approximation.
 * @param[in] work: Working array.
 * @param[in] lwork: Size of `work` array.
 * @param[in] iwork: Temporary integer array.
 * */
{
    int oversample = starsh_params.rsvd_oversample;
    int mn = nrows < ncols ? nrows : ncols;
    int mn2 = maxrank+oversample;
    if(mn2 > mn)
        mn2 = mn;
    size_t svdqr_lwork = (4*(size_t)mn2+7)*mn2;
    if(svdqr_lwork < 3*ncols+1)
        svdqr_lwork = 3*ncols+1;
    double *R, *tau, *svd_U, *svd_S, *svd_V, *svdqr_work;
    R = work;
    tau = R+(size_t)ncols*mn2;
    svd_U = tau+mn;
    svd_S = svd_U+(size_t)mn2*mn2;
    svd_V = svd_S+mn2;
    svdqr_work = svd_V+(size_t)ncols*mn2;
    // Set pivots for GEQP3 to zeros
    for(int i = 0; i < ncols; i++)
        iwork[i] = 0;
    // Call GEQP3
    LAPACKE_dgeqp3_work(LAPACK_COL_MAJOR, nrows, ncols, D, nrows, iwork,
            tau, svdqr_work, svdqr_lwork);
    // Copy R factor to V
    for(size_t i = 0; i < ncols; i++)
    {
        size_t j = iwork[i]-1;
        size_t k = i < mn2 ? i+1 : mn2;
        cblas_dcopy(k, D+i*nrows, 1, R+j*mn2, 1);
        for(size_t l = k; l < mn2; l++)
            R[j*mn2+l] = 0.;
    }
    // Get factor Q
    LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, nrows, mn2, mn2, D, nrows, tau,
            svdqr_work, svdqr_lwork);
    // Get SVD of corresponding matrix to reduce rank
    LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', mn2, ncols, R, mn2, svd_S,
            svd_U, mn2, svd_V, mn2, svdqr_work, svdqr_lwork, iwork);
    // Get rank, corresponding to given error tolerance
    *rank = starsh_dense_dsvfr(mn2, svd_S, tol);
    if(*rank < mn/2 && *rank <= maxrank)
    // If far-field block is low-rank
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, *rank,
                mn2, 1.0, D, nrows, svd_U, mn2, 0.0, U, nrows);
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
