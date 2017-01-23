#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "stars.h"
#include "misc.h"

int starsh_blrm__dqp3(STARSH_blrm **M, STARSH_blrf *F, int maxrank,
        int oversample, double tol, int onfly)
// Double precision Tile Low-Rank geSDD approximation
{
    STARSH_problem *P = F->problem;
    block_kernel kernel = P->kernel;
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near;
    // Following values default to given block low-rank format F, but they are
    // changed when there are false far-field blocks.
    size_t new_nblocks_far = nblocks_far, new_nblocks_near = nblocks_near;
    int *block_far = F->block_far, *block_near = F->block_near;
    // Places to store low-rank factors, dense blocks and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank = NULL;
    // Init buffers to store low-rank factors of far-field blocks if needed
    if(nblocks_far > 0)
    {
        STARSH_MALLOC(far_U, nblocks_far);
        STARSH_MALLOC(far_V, nblocks_far);
        STARSH_MALLOC(far_rank, nblocks_far);
    }
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster, *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    // Work variables
    int info;
    size_t bi, bj = 0;
    // Simple cycle over all far-field admissible blocks
    for(bi = 0; bi < nblocks_far; bi++)
    {
        // Get indexes of corresponding block row and block column
        int i = block_far[2*bi];
        int j = block_far[2*bi+1];
        // Get corresponding sizes and minimum of them
        int nrows = RC->size[i];
        int ncols = CC->size[j];
        int mn = nrows < ncols ? nrows : ncols;
        //int mn2 = mn < maxrank ? mn : maxrank;
        int mn2 = maxrank+oversample;
        if(mn2 > mn)
            mn2 = mn;
        // Get size of temporary arrays
        size_t lwork = 3*ncols+1, lwork_sdd = (4*(size_t)mn2+7)*mn2;
        if(lwork_sdd > lwork)
            lwork = lwork_sdd;
        size_t liwork = ncols, liwork_sdd = 8*mn2;
        if(liwork_sdd > liwork)
            liwork = liwork_sdd;
        double *D, *R, *work, *U, *V, *tau, *svd_U, *svd_S, *svd_V;
        int *iwork, *ipiv;
        // Allocate temporary arrays
        STARSH_MALLOC(D, (size_t)nrows*(size_t)ncols);
        STARSH_MALLOC(R, (size_t)mn2*(size_t)ncols);
        STARSH_MALLOC(iwork, liwork);
        ipiv = iwork;
        STARSH_MALLOC(work, lwork);
        STARSH_MALLOC(tau, mn);
        STARSH_MALLOC(svd_U, (size_t)mn2*(size_t)mn2);
        STARSH_MALLOC(svd_S, mn2);
        STARSH_MALLOC(svd_V, (size_t)mn2*(size_t)ncols);
        // Compute elements of a block
        kernel(nrows, ncols, RC->pivot+RC->start[i], CC->pivot+CC->start[j],
                RD, CD, D);
        // Set pivots for GEQP3 to zeros
        for(int k = 0; k < ncols; k++)
            ipiv[k] = 0;
        // Call GEQP3
        LAPACKE_dgeqp3_work(LAPACK_COL_MAJOR, nrows, ncols, D, nrows, ipiv,
                tau, work, lwork);
        // Copy R factor to V
        for(size_t k = 0; k < ncols; k++)
        {
            size_t kk = ipiv[k]-1;
            size_t l = k < mn2 ? k+1 : mn2;
            cblas_dcopy(l, D+k*nrows, 1, R+kk*mn2, 1);
            for(size_t ll = l; ll < mn2; ll++)
                R[kk*mn2+ll] = 0.;
        }
        // Get factor Q
        LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, nrows, mn2, mn2, D, nrows, tau,
                work, lwork);
        free(tau);
        // Get SVD of corresponding matrix to reduce rank
        LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', mn2, ncols, R, mn2, svd_S,
                svd_U, mn2, svd_V, mn2, work, lwork, iwork);
        free(work);
        free(iwork);
        // Get rank, corresponding to given error tolerance
        int rank = starsh__dsvfr(mn2, svd_S, tol);
        if(rank < mn/2 && rank <= maxrank)
        // If far-field block is low-rank
        {
            far_rank[bi] = rank;
            int shapeU[2] = {nrows, rank}, shapeV[2] = {ncols, rank};
            array_new(far_U+bi, 2, shapeU, 'd', 'F');
            array_new(far_V+bi, 2, shapeV, 'd', 'F');
            U = far_U[bi]->data;
            V = far_V[bi]->data;
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, rank,
                    mn2, 1.0, D, nrows, svd_U, mn2, 0.0, U, nrows);
            for(size_t k = 0; k < rank; k++)
            {
                //cblas_dcopy(nrows, U+k*nrows, 1, U2+k*nrows, 1);
                cblas_dcopy(ncols, svd_V+k, mn2, V+k*ncols, 1);
                cblas_dscal(ncols, svd_S[k], V+k*ncols, 1);
            }
        }
        else
        // If far-field block is dense, although it was initially assumed
        // to be low-rank. Let denote such a block as false far-field block
        {
            far_rank[bi] = -1;
            far_U[bi] = NULL;
            far_V[bi] = NULL;
        }
        // Free temporary arrays
        free(D);
        free(R);
        free(svd_U);
        free(svd_S);
        free(svd_V);
    }
    // Get number of false far-field blocks
    size_t nblocks_false_far = 0;
    size_t *false_far = NULL;
    for(bi = 0; bi < nblocks_far; bi++)
        if(far_rank[bi] == -1)
            nblocks_false_far++;
    if(nblocks_false_far > 0)
    {
        // IMPORTANT: `false_far` must to be in ascending order for later code
        // to work normally
        STARSH_MALLOC(false_far, nblocks_false_far);
        bj = 0;
        for(bi = 0; bi < nblocks_far; bi++)
            if(far_rank[bi] == -1)
                false_far[bj++] = bi;
    }
    // Update lists of far-field and near-field blocks using previously
    // generated list of false far-field blocks
    if(nblocks_false_far > 0)
    {
        // Update list of near-field blocks
        new_nblocks_near = nblocks_near+nblocks_false_far;
        STARSH_MALLOC(block_near, 2*new_nblocks_near);
        // At first get all near-field blocks, assumed to be dense
        for(bi = 0; bi < 2*nblocks_near; bi++)
            block_near[bi] = F->block_near[bi];
        // Add false far-field blocks
        for(bi = 0; bi < nblocks_false_far; bi++)
        {
            size_t bj = false_far[bi];
            block_near[2*(bi+nblocks_near)] = F->block_far[2*bj];
            block_near[2*(bi+nblocks_near)+1] = F->block_far[2*bj+1];
        }
        // Update list of far-field blocks
        new_nblocks_far = nblocks_far-nblocks_false_far;
        if(new_nblocks_far > 0)
        {
            STARSH_MALLOC(block_far, 2*new_nblocks_far);
            bj = 0;
            for(bi = 0; bi < nblocks_far; bi++)
            {
                // `false_far` must be in ascending order for this to work
                if(false_far[bj] == bi)
                {
                    bj++;
                }
                else
                {
                    block_far[2*(bi-bj)] = F->block_far[2*bi];
                    block_far[2*(bi-bj)+1] = F->block_far[2*bi+1];
                }
            }
        }
        // Update format by creating new format
        STARSH_blrf *F2;
        info = starsh_blrf_new(&F2, P, F->symm, RC, CC, new_nblocks_far,
                block_far, new_nblocks_near, block_near, F->type);
        // Swap internal data of formats and free unnecessary data
        STARSH_blrf tmp_blrf = *F;
        *F = *F2;
        *F2 = tmp_blrf;
        STARSH_WARNING("`F` was modified due to false far-field blocks");
        info = starsh_blrf_free(F2);
        if(info != 0)
            return info;
    }
    // Compute near-field blocks if needed
    if(onfly == 0 && new_nblocks_near > 0)
    {
        STARSH_MALLOC(near_D, new_nblocks_near);
        // For each near-field block compute its elements
        for(bi = 0; bi < new_nblocks_near; bi++)
        {
            // Get indexes of corresponding block row and block column
            int i = block_near[2*bi];
            int j = block_near[2*bi+1];
            // Get corresponding sizes and minimum of them
            int nrows = RC->size[i];
            int ncols = CC->size[j];
            int shape[2] = {nrows, ncols};
            array_new(near_D+bi, 2, shape, 'd', 'F');
            kernel(nrows, ncols, RC->pivot+RC->start[i],
                    CC->pivot+CC->start[j], RD, CD, near_D[bi]->data);
        }
    }
    // Change sizes of far_rank, far_U and far_V if there were false
    // far-field blocks
    if(nblocks_false_far > 0 && new_nblocks_far > 0)
    {
        bj = 0;
        for(bi = 0; bi < nblocks_far; bi++)
        {
            if(false_far[bj] == bi)
                bj++;
            else
            {
                far_U[bi-bj] = far_U[bi];
                far_V[bi-bj] = far_V[bi];
                far_rank[bi-bj] = far_rank[bi];
            }
        }
        STARSH_REALLOC(far_rank, new_nblocks_far);
        STARSH_REALLOC(far_U, new_nblocks_far);
        STARSH_REALLOC(far_V, new_nblocks_far);
    }
    // If all far-field blocks are false, then dealloc buffers
    if(new_nblocks_far == 0 && nblocks_far > 0)
    {
        block_far = NULL;
        free(far_rank);
        far_rank = NULL;
        free(far_U);
        far_U = NULL;
        free(far_V);
        far_V = NULL;
    }
    // Dealloc list of false far-field blocks if it is not empty
    if(nblocks_false_far > 0)
        free(false_far);
    // Finish with creating instance of Block Low-Rank Matrix with given
    // buffers
    return starsh_blrm_new(M, F, far_rank, far_U, far_V, onfly, near_D,
            NULL, NULL, NULL, '2');
}

