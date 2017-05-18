/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * @file backends/sequential/kernels/dsdd.c
 * @version 1.0.0.2
 * @author Aleksandr Mikhalev
 * @date 16 May 2017
 * */

#include "common.h"
#include "starsh.h"

void starsh_kernel_dsdd(int nrows, int ncols, double *D, double *U, double *V,
        int *rank, int maxrank, int oversample, double tol, double *work,
        int lwork, int *iwork)
//! DGESDD approximation of a dense double precision matrix.
/*! @ingroup approximations
 * @param[in] nrows: Number of rows of a matrix.
 * @param[in] ncols: Number of columns of a matrix.
 * @param[in,out] D: Pointer to dense matrix.
 * @param[out] U: Pointer to low-rank factor `U`.
 * @param[out] V: Pointer to low-rank factor `V`.
 * @param[out] rank: Address of rank variable.
 * @param[in] maxrank: Maximum possible rank.
 * @param[in] oversample: Ignored.
 * @param[in] tol: Relative error for approximation.
 * @param[in] work: Working array.
 * @param[in] lwork: Size of `work` array.
 * @param[in] iwork: Temporary integer array.
 * */
{
    (void)oversample;
    int mn = nrows < ncols ? nrows : ncols;
    size_t svd_lwork = (4*(size_t)mn+7)*mn;
    double *svd_U, *svd_S, *svd_V, *svd_work;
    svd_U = work;
    svd_S = svd_U+(size_t)nrows*mn;
    svd_V = svd_S+mn;
    svd_work = svd_V+(size_t)ncols*mn;
    // Get SVD via GESDD function of LAPACK
    LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', nrows, ncols, D, nrows,
            svd_S, svd_U, nrows, svd_V, mn, svd_work, svd_lwork, iwork);
    // Get rank, corresponding to given error tolerance
    *rank = starsh__dsvfr(mn, svd_S, tol);
    if(*rank < mn/2 && *rank <= maxrank)
    // If far-field block is low-rank
    {
        for(size_t i = 0; i < *rank; i++)
        {
            cblas_dcopy(nrows, svd_U+i*nrows, 1, U+i*nrows, 1);
            cblas_dcopy(ncols, svd_V+i, mn, V+i*ncols, 1);
            cblas_dscal(ncols, svd_S[i], V+i*ncols, 1);
        }
    }
    else
    // If far-field block is dense, although it was initially assumed
    // to be low-rank. Let denote such a block as false far-field block
        *rank = -1;
}
