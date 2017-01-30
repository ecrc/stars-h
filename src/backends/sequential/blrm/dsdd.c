/*!
 * @file dsdd.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "stars.h"
#include "misc.h"

/*!
 * Approximate H-matrix by sdd procedure for every block.
 *
 * Uses DGESDD for each block. Complexity is very high!
 *
 * @param[out] M text
 * @param[in,out] F text
 * @param[in] tol text
 * @param[in] onfly text
 * @result error code or 0 if everything is OK
 */
int starsh_blrm__dsdd(STARSH_blrm **M, STARSH_blrf *F, int maxrank,
        int oversample, double tol, int onfly)
{
    STARSH_problem *P = F->problem;
    STARSH_kernel kernel = P->kernel;
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near;
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster, *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    // Following values default to given block low-rank format F, but they are
    // changed when there are false far-field blocks.
    size_t new_nblocks_far = nblocks_far, new_nblocks_near = nblocks_near;
    int *block_far = F->block_far, *block_near = F->block_near;
    // Places to store low-rank factors, dense blocks and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank = NULL;
    double *alloc_U = NULL, *alloc_V = NULL, *alloc_D = NULL;
    size_t offset_U = 0, offset_V = 0, offset_D = 0;
    size_t bi, bj = 0;
    // Init buffers to store low-rank factors of far-field blocks if needed
    if(nblocks_far > 0)
    {
        STARSH_MALLOC(far_U, nblocks_far);
        STARSH_MALLOC(far_V, nblocks_far);
        STARSH_MALLOC(far_rank, nblocks_far);
        size_t size_U = 0, size_V = 0;
        // Simple cycle over all far-field blocks
        for(bi = 0; bi < nblocks_far; bi++)
        {
            // Get indexes of corresponding block row and block column
            int i = block_far[2*bi];
            int j = block_far[2*bi+1];
            // Get corresponding sizes and minimum of them
            size_t nrows = RC->size[i], ncols = CC->size[j];
            size_U += nrows*maxrank;
            size_V += ncols*maxrank;
        }
        STARSH_MALLOC(alloc_U, size_U);
        STARSH_MALLOC(alloc_V, size_V);
        for(bi = 0; bi < nblocks_far; bi++)
        {
            // Get indexes of corresponding block row and block column
            int i = block_far[2*bi];
            int j = block_far[2*bi+1];
            // Get corresponding sizes and minimum of them
            size_t nrows = RC->size[i], ncols = CC->size[j];
            int shape_U[] = {nrows, maxrank};
            int shape_V[] = {ncols, maxrank};
            double *U = alloc_U+offset_U, *V = alloc_V+offset_V;
            offset_U += nrows*maxrank;
            offset_V += ncols*maxrank;
            array_from_buffer(far_U+bi, 2, shape_U, 'd', 'F', U);
            array_from_buffer(far_V+bi, 2, shape_V, 'd', 'F', V);
        }
        offset_U = 0;
        offset_V = 0;
    }
    // Work variables
    int info;
    // Simple cycle over all far-field admissible blocks
#pragma omp parallel for schedule(dynamic,1)
    for(bi = 0; bi < nblocks_far; bi++)
    {
        int info;
        // Get indexes of corresponding block row and block column
        int i = block_far[2*bi];
        int j = block_far[2*bi+1];
        // Get corresponding sizes and minimum of them
        int nrows = RC->size[i];
        int ncols = CC->size[j];
        int mn = nrows > ncols ? ncols : nrows;
        // Get size of temporary arrays
        size_t lmn = mn, lwork = (4*lmn+8+nrows+ncols)*lmn, liwork = 8*lmn;
        double *D, *work;
        int *iwork;
        size_t D_size = (size_t)nrows*(size_t)ncols;
        // Allocate temporary arrays
        STARSH_PMALLOC(D, D_size, info);
        STARSH_PMALLOC(work, lwork, info);
        STARSH_PMALLOC(iwork, liwork, info);
        // Compute elements of a block
        kernel(nrows, ncols, RC->pivot+RC->start[i], CC->pivot+CC->start[j],
                RD, CD, D);
        starsh_kernel_dsdd(nrows, ncols, D, far_U[bi]->data, far_V[bi]->data,
                far_rank+bi, maxrank, oversample, tol, work, lwork, iwork);
        // Free temporary arrays
        free(D);
        free(work);
        free(iwork);
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
        size_t size_D = 0;
        // Simple cycle over all near-field blocks
        for(bi = 0; bi < new_nblocks_near; bi++)
        {
            // Get indexes of corresponding block row and block column
            int i = block_near[2*bi];
            int j = block_near[2*bi+1];
            // Get corresponding sizes and minimum of them
            size_t nrows = RC->size[i];
            size_t ncols = CC->size[j];
            // Update size_D
            size_D += nrows*ncols;
        }
        STARSH_MALLOC(alloc_D, size_D);
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
            double *D = alloc_D+offset_D;
            array_from_buffer(near_D+bi, 2, shape, 'd', 'F', D);
            offset_D += near_D[bi]->size;
            kernel(nrows, ncols, RC->pivot+RC->start[i],
                    CC->pivot+CC->start[j], RD, CD, D);
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
        //STARSH_REALLOC(alloc_U, offset_U);
        //STARSH_REALLOC(alloc_V, offset_V);
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
        free(alloc_U);
        alloc_U = NULL;
        free(alloc_V);
        alloc_V = NULL;
    }
    // Dealloc list of false far-field blocks if it is not empty
    if(nblocks_false_far > 0)
        free(false_far);
    // Finish with creating instance of Block Low-Rank Matrix with given
    // buffers
    return starsh_blrm_new(M, F, far_rank, far_U, far_V, onfly, near_D,
            alloc_U, alloc_V, alloc_D, '1');
}
