/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/sequential/dense/zrsdd.c
 * @version 0.3.0
 * @author Rabab Alomairy
 * @author Kadir Akbudak
 * @author Aleksandr Mikhalev
 * @date 2020-06-07
 * */

#include "common.h"
#include "starsh.h"



double* starsh_dense_zlrrsdd(int nrows, int ncols, double _Complex  *D, int ldD, double _Complex  *U,
        int ldU, double _Complex  *V, int ldV, int *rank, int maxrank, int oversample,
        double tol, double _Complex  *work, int lwork, int *iwork)
//! Randomized SVD approximation of a dense double precision matrix.
/*! This function calls LAPACK and BLAS routines, so integer types are int
 * instead of @ref STARSH_int.
 *
 * @param[in] nrows: Number of rows of a matrix.
 * @param[in] ncols: Number of columns of a matrix.
 * @param[in,out] D: Pointer to dense matrix.
 * @param[in] ldD: leading dimensions of `D`.
 * @param[out] U: Pointer to low-rank factor `U`.
 * @param[in] ldU: leading dimensions of `U`.
 * @param[out] V: Pointer to low-rank factor `V`.
 * @param[in] ldV: leading dimensions of `V`.
 * @param[out] rank: Address of rank variable.
 * @param[in] maxrank: Maximum possible rank.
 * @param[in] oversample: Size of oversampling subset.
 * @param[in] tol: Relative error for approximation.
 * @param[in] work: Working array.
 * @param[in] lwork: Size of `work` array.
 * @param[in] iwork: Temporary integer array.
 * */
{
    int mn = nrows < ncols ? nrows : ncols;
    int mn2 = maxrank+oversample;
    int i;
    if(mn2 > mn)
        mn2 = mn;
    //size_t svdqr_lwork = (4*mn2+7)*mn2;
    //if(svdqr_lwork < ncols)
    //    svdqr_lwork = ncols;
    double _Complex *X, *Q, *tau, *svd_U, *svd_V, *svdqr_work;
    double *svd_S, *svd_rwork;

    int mx = nrows>ncols?nrows:ncols;
    size_t s1 = 5*mn*mn + 5*mn;
    size_t s2 = 2*mx*mn + 2*mn*mn + mn;
    size_t lrwork = s1>s2?s1:s2;
    svd_rwork = malloc(sizeof(*svd_rwork) * lrwork);
    //double *X, *Q, *tau, *svd_U, *svd_S, *svd_V, *svdqr_work;
    X = work;
    Q = X+(size_t)ncols*mn2;
    svd_U = Q+(size_t)nrows*mn2;
    svd_S = svd_U+(size_t)mn2*mn2;
    tau = svd_S;
    svd_V = svd_S+mn2;
    svdqr_work = svd_V+ncols*mn2;
    int svdqr_lwork = lwork-(size_t)mn2*(2*ncols+nrows+mn2+1);
    int iseed[4] = {0, 0, 0, 1};
    double _Complex zero = (double _Complex) 0.0;
    double _Complex one = (double _Complex) 1.0;
    // Generate random matrix X
    LAPACKE_zlarnv_work(3, iseed, nrows*mn2, X);
    // Multiply by random matrix
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, mn2,
            ncols, &one, D, ldD, X, ncols, &zero, Q, nrows);
    // Get Q factor of QR factorization
    LAPACKE_zgeqrf_work(LAPACK_COL_MAJOR, nrows, mn2, Q, nrows, tau,
            svdqr_work, svdqr_lwork);
    LAPACKE_zungqr_work(LAPACK_COL_MAJOR, nrows, mn2, mn2, Q, nrows, tau,
            svdqr_work, svdqr_lwork);
    // Multiply Q by initial matrix
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, mn2, ncols,
            nrows, &one, Q, nrows, D, ldD, &zero, X, mn2);
    // Get SVD of result to reduce rank
    int info = LAPACKE_zgesdd_work(LAPACK_COL_MAJOR, 'S', mn2, ncols, X, mn2,
            svd_S, svd_U, mn2, svd_V, mn2, svdqr_work, svdqr_lwork, svd_rwork, iwork);
    if(info != 0) {
        printf("%s %d: LAPACKE_dgesdd_work info=%d", __FILE__, __LINE__, info);
        exit(-1);
    }
    // Get rank, corresponding to given error tolerance
    *rank = starsh_dense_dsvfr(mn2, svd_S, tol);
    //printf("%s %d: info:%d rank:%d\n", __FILE__, __LINE__, info, *rank);
    if(info == 0 && *rank <= maxrank)
    // If far-field block is low-rank
    {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, *rank,
                mn2, &one, Q, nrows, svd_U, mn2, &zero, U, ldU);
        for(i = 0; i < *rank; i++)
        {
            cblas_zcopy(ncols, svd_V+i, mn2, V+i*(size_t)ldV, 1);
            double _Complex __svd_Si = (double _Complex) svd_S[i];
            cblas_zscal(ncols, &__svd_Si, V+i*(size_t)ldV, 1);
        }
    }
    else
    // If far-field block is dense, although it was initially assumed
    // to be low-rank. Let denote such a block as false far-field block
        *rank = -1;
    free(svd_rwork);
    return svd_S;
}

