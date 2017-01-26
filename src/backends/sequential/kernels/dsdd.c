#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "stars.h"

void starsh_kernel_dsdd(int nrows, int ncols, double *D, Array *U, Array *V,
        int *rank, int maxrank, int oversample, double tol, double *work,
        int lwork, int *iwork)
{
    (void)oversample;
    int mn = nrows < ncols ? nrows : ncols;
    size_t svd_lwork = (4*(size_t)mn+7)*mn;
    double *svd_U, *svd_S, *svd_V, *svd_work, *out_U, *out_V;
    svd_U = work;
    svd_S = svd_U+(size_t)nrows*mn;
    svd_V = svd_S+mn;
    svd_work = svd_V+(size_t)ncols*mn;
    out_U = U->data;
    out_V = V->data;
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
            cblas_dcopy(nrows, svd_U+i*nrows, 1, out_U+i*nrows, 1);
            cblas_dcopy(ncols, svd_V+i, mn, out_V+i*ncols, 1);
            cblas_dscal(ncols, svd_S[i], out_V+i*ncols, 1);
        }
        U->shape[1] = *rank;
        V->shape[1] = *rank;
    }
    else
    // If far-field block is dense, although it was initially assumed
    // to be low-rank. Let denote such a block as false far-field block
        *rank = -1;
}
