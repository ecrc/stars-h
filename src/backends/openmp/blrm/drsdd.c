/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/openmp/blrm/drsdd.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"

int starsh_blrm__drsdd_omp(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly)
//! Approximate each tile by randomized SVD.
/*!
 * @param[out] matrix: Address of pointer to @ref STARSH_blrm object.
 * @param[in] format: Block low-rank format.
 * @param[in] maxrank: Maximum possible rank.
 * @param[in] tol: Relative error tolerance.
 * @param[in] onfly: Whether not to store dense blocks.
 * @ingroup blrm
 * */
{
    STARSH_blrf *F = format;
    STARSH_problem *P = F->problem;
    STARSH_kernel *kernel = P->kernel;
    STARSH_int nblocks_far = F->nblocks_far;
    STARSH_int nblocks_near = F->nblocks_near;
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster;
    STARSH_cluster *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    // Following values default to given block low-rank format F, but they are
    // changed when there are false far-field blocks.
    STARSH_int new_nblocks_far = nblocks_far;
    STARSH_int new_nblocks_near = nblocks_near;
    STARSH_int *block_far = F->block_far;
    STARSH_int *block_near = F->block_near;
    // Places to store low-rank factors, dense blocks and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank = NULL;
    double *alloc_U = NULL, *alloc_V = NULL, *alloc_D = NULL;
    size_t offset_U = 0, offset_V = 0, offset_D = 0;
    STARSH_int bi, bj = 0;
    double drsdd_time = 0, kernel_time = 0;
    int BAD_TILE = 0;
    const int oversample = starsh_params.oversample;
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
            STARSH_int i = block_far[2*bi];
            STARSH_int j = block_far[2*bi+1];
            // Get corresponding sizes and minimum of them
            size_U += RC->size[i];
            size_V += CC->size[j];
        }
        size_U *= maxrank;
        size_V *= maxrank;
        STARSH_MALLOC(alloc_U, size_U);
        STARSH_MALLOC(alloc_V, size_V);
        for(bi = 0; bi < nblocks_far; bi++)
        {
            // Get indexes of corresponding block row and block column
            STARSH_int i = block_far[2*bi];
            STARSH_int j = block_far[2*bi+1];
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
        // Get indexes of corresponding block row and block column
        STARSH_int i = block_far[2*bi];
        STARSH_int j = block_far[2*bi+1];
        // Get corresponding sizes and minimum of them
        int nrows = RC->size[i];
        int ncols = CC->size[j];
        if(nrows != ncols && BAD_TILE == 0)
        {
            #pragma omp critical
            BAD_TILE = 1;
            STARSH_WARNING("This was only tested on square tiles, error of "
                    "approximation may be much higher, than demanded");
        }
        int mn = nrows < ncols ? nrows : ncols;
        int mn2 = maxrank+oversample;
        if(mn2 > mn)
            mn2 = mn;
        // Get size of temporary arrays
        int lwork = ncols, lwork_sdd = (4*mn2+7)*mn2;
        if(lwork_sdd > lwork)
            lwork = lwork_sdd;
        lwork += (size_t)mn2*(2*ncols+nrows+mn2+1);
        int liwork = 8*mn2;
        double *D, *work;
        int *iwork;
        int info;
        // Allocate temporary arrays
        STARSH_PMALLOC(D, (size_t)nrows*(size_t)ncols, info);
        STARSH_PMALLOC(iwork, liwork, info);
        STARSH_PMALLOC(work, lwork, info);
        // Compute elements of a block
        double time0 = omp_get_wtime();
        kernel(nrows, ncols, RC->pivot+RC->start[i], CC->pivot+CC->start[j],
                RD, CD, D, nrows);
        double time1 = omp_get_wtime();
        starsh_dense_dlrrsdd(nrows, ncols, D, nrows, far_U[bi]->data, nrows,
                far_V[bi]->data, ncols, far_rank+bi, maxrank, oversample, tol,
                work, lwork, iwork);
        double time2 = omp_get_wtime();
        #pragma omp critical
        {
            drsdd_time += time2-time1;
            kernel_time += time1-time0;
        }
        // Free temporary arrays
        free(D);
        free(work);
        free(iwork);
    }
    // Get number of false far-field blocks
    STARSH_int nblocks_false_far = 0;
    STARSH_int *false_far = NULL;
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
        #pragma omp parallel for schedule(static)
        for(bi = 0; bi < 2*nblocks_near; bi++)
            block_near[bi] = F->block_near[bi];
        // Add false far-field blocks
        #pragma omp parallel for schedule(static)
        for(bi = 0; bi < nblocks_false_far; bi++)
        {
            STARSH_int bj = false_far[bi];
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
                if(bj < nblocks_false_far && false_far[bj] == bi)
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
        info = starsh_blrf_new_from_coo(&F2, P, F->symm, RC, CC,
                new_nblocks_far, block_far, new_nblocks_near, block_near,
                F->type);
        // Swap internal data of formats and free unnecessary data
        STARSH_blrf tmp_blrf = *F;
        *F = *F2;
        *F2 = tmp_blrf;
        STARSH_WARNING("`F` was modified due to false far-field blocks");
        starsh_blrf_free(F2);
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
            STARSH_int i = block_near[2*bi];
            STARSH_int j = block_near[2*bi+1];
            // Get corresponding sizes and minimum of them
            size_t nrows = RC->size[i];
            size_t ncols = CC->size[j];
            // Update size_D
            size_D += nrows*ncols;
        }
        STARSH_MALLOC(alloc_D, size_D);
        // For each near-field block compute its elements
        #pragma omp parallel for schedule(dynamic,1)
        for(bi = 0; bi < new_nblocks_near; bi++)
        {
            // Get indexes of corresponding block row and block column
            STARSH_int i = block_near[2*bi];
            STARSH_int j = block_near[2*bi+1];
            // Get corresponding sizes and minimum of them
            int nrows = RC->size[i];
            int ncols = CC->size[j];
            int shape[2] = {nrows, ncols};
            double *D;
            #pragma omp critical
            {
                D = alloc_D+offset_D;
                array_from_buffer(near_D+bi, 2, shape, 'd', 'F', D);
                offset_D += near_D[bi]->size;
            }
            double time0 = omp_get_wtime();
            kernel(nrows, ncols, RC->pivot+RC->start[i],
                    CC->pivot+CC->start[j], RD, CD, D, nrows);
            double time1 = omp_get_wtime();
            #pragma omp critical
            kernel_time += time1-time0;
        }
    }
    // Change sizes of far_rank, far_U and far_V if there were false
    // far-field blocks
    if(nblocks_false_far > 0 && new_nblocks_far > 0)
    {
        bj = 0;
        for(bi = 0; bi < nblocks_far; bi++)
        {
            if(far_rank[bi] == -1)
                bj++;
            else
            {
                int shape_U[2] = {far_U[bi]->shape[0], far_rank[bi]};
                int shape_V[2] = {far_V[bi]->shape[0], far_rank[bi]};
                array_from_buffer(far_U+bi-bj, 2, shape_U, 'd', 'F',
                        far_U[bi]->data);
                array_from_buffer(far_V+bi-bj, 2, shape_V, 'd', 'F',
                        far_V[bi]->data);
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
    //STARSH_WARNING("DRSDD kernel total time: %e secs", drsdd_time);
    //STARSH_WARNING("MATRIX kernel total time: %e secs", kernel_time);
    return starsh_blrm_new(matrix, F, far_rank, far_U, far_V, onfly, near_D,
            alloc_U, alloc_V, alloc_D, '1');
}

