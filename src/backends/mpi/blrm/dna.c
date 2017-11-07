/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/mpi/blrm/dna.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-mpi.h"

int starsh_blrm__dna_mpi(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly)
//! Simply compute matrix without any approximation.
/*! @ingroup blrm
 * @param[out] matrix: Address of pointer to `STARSH_blrm` object.
 * @param[in] format: Block low-rank format.
 * @param[in] maxrank: Maximum possible rank.
 * @param[in] tol: Relative error tolerance.
 * @param[in] onfly: Whether not to store dense blocks.
 * */
{
    STARSH_blrf *F = format;
    STARSH_problem *P = F->problem;
    STARSH_kernel *kernel = P->kernel;
    STARSH_int nblocks_far = F->nblocks_far;
    STARSH_int nblocks_near = F->nblocks_near;
    STARSH_int nblocks_far_local = F->nblocks_far_local;
    STARSH_int nblocks_near_local = F->nblocks_near_local;
    // Shortcuts to information about clusters
    STARSH_cluster *RC = F->row_cluster;
    STARSH_cluster *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    // Following values default to given block low-rank format F, but they are
    // changed when there are false far-field blocks.
    STARSH_int new_nblocks_far = F->nblocks_far;
    STARSH_int new_nblocks_near = F->nblocks_near;
    STARSH_int new_nblocks_far_local = F->nblocks_far_local;
    STARSH_int new_nblocks_near_local = F->nblocks_near_local;
    STARSH_int *block_far = F->block_far;
    STARSH_int *block_near = F->block_near;
    STARSH_int *block_far_local = F->block_far_local;
    STARSH_int *block_near_local = F->block_near_local;
    // Places to store low-rank factors, dense blocks and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank = NULL;
    double *alloc_U = NULL, *alloc_V = NULL, *alloc_D = NULL;
    size_t offset_U = 0, offset_V = 0, offset_D = 0;
    STARSH_int lbi, lbj, bi, bj = 0;
    double drsdd_time = 0, kernel_time = 0;
    int BAD_TILE = 0;
    // Init buffers to store low-rank factors of far-field blocks if needed
    if(nblocks_far > 0)
    {
        STARSH_MALLOC(far_U, nblocks_far_local);
        STARSH_MALLOC(far_V, nblocks_far_local);
        STARSH_MALLOC(far_rank, nblocks_far_local);
        size_t size_U = 0, size_V = 0;
        // Simple cycle over all far-field blocks
        for(lbi = 0; lbi < nblocks_far_local; lbi++)
        {
            STARSH_int bi = block_far_local[lbi];
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
        for(lbi = 0; lbi < nblocks_far_local; lbi++)
        {
            STARSH_int bi = block_far_local[lbi];
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
            array_from_buffer(far_U+lbi, 2, shape_U, 'd', 'F', U);
            array_from_buffer(far_V+lbi, 2, shape_V, 'd', 'F', V);
        }
        offset_U = 0;
        offset_V = 0;
    }
    // Work variables
    int info;
    // Simple cycle over all far-field admissible blocks
    // Since this is fake low-rank approximation, every tile is dense
    #pragma omp parallel for schedule(static)
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
        far_rank[lbi] = -1;
    /*
    #pragma omp parallel for schedule(dynamic, 1)
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
    {
        size_t bi = block_far_local[lbi];
        // Get indexes of corresponding block row and block column
        int i = block_far[2*bi];
        int j = block_far[2*bi+1];
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
        size_t lwork = ncols, lwork_sdd = (4*mn2+7)*mn2;
        if(lwork_sdd > lwork)
            lwork = lwork_sdd;
        lwork += (size_t)mn2*(2*ncols+nrows+mn2+1);
        size_t liwork = 8*mn2;
        double *D, *work;
        int *iwork;
        int info;
        // Allocate temporary arrays
        STARSH_PMALLOC(D, (size_t)nrows*(size_t)ncols, info);
        //STARSH_PMALLOC(iwork, liwork, info);
        //STARSH_PMALLOC(work, lwork, info);
        // Compute elements of a block
        double time0 = omp_get_wtime();
        kernel(nrows, ncols, RC->pivot+RC->start[i], CC->pivot+CC->start[j],
                RD, CD, D);
        double time1 = omp_get_wtime();
        starsh_kernel_dna(nrows, ncols, D, far_U[lbi]->data,
                far_V[lbi]->data, far_rank+lbi, maxrank, oversample, tol, work,
                lwork, iwork);
        double time2 = omp_get_wtime();
        #pragma omp critical
        {
            drsdd_time += time2-time1;
            kernel_time += time1-time0;
        }
        // Free temporary arrays
        free(D);
        //free(work);
        //free(iwork);
    }
    */
    // Get number of false far-field blocks
    STARSH_int nblocks_false_far_local = 0;
    STARSH_int *false_far_local = NULL;
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
        if(far_rank[lbi] == -1)
            nblocks_false_far_local++;
    if(nblocks_false_far_local > 0)
    {
        // IMPORTANT: `false_far` and `false_far_local` must be in
        // ascending order for later code to work normally
        STARSH_MALLOC(false_far_local, nblocks_false_far_local);
        lbj = 0;
        for(lbi = 0; lbi < nblocks_far_local; lbi++)
            if(far_rank[lbi] == -1)
                false_far_local[lbj++] = block_far_local[lbi];
    }
    // Sync list of all false far-field blocks
    STARSH_int nblocks_false_far = 0;
    STARSH_int int_nblocks_false_far_local = nblocks_false_far_local;
    int *mpi_recvcount, *mpi_offset;
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    STARSH_MALLOC(mpi_recvcount, mpi_size);
    STARSH_MALLOC(mpi_offset, mpi_size);
    MPI_Allgather(&int_nblocks_false_far_local, 1, MPI_INT, mpi_recvcount,
            1, MPI_INT, MPI_COMM_WORLD);
    for(bi = 0; bi < mpi_size; bi++)
        nblocks_false_far += mpi_recvcount[bi];
    mpi_offset[0] = 0;
    for(bi = 1; bi < mpi_size; bi++)
        mpi_offset[bi] = mpi_offset[bi-1]+mpi_recvcount[bi-1];
    size_t *false_far = NULL;
    if(nblocks_false_far > 0)
        STARSH_MALLOC(false_far, nblocks_false_far);
    MPI_Allgatherv(false_far_local, nblocks_false_far_local, my_MPI_SIZE_T,
            false_far, mpi_recvcount, mpi_offset, my_MPI_SIZE_T,
            MPI_COMM_WORLD);
    free(mpi_recvcount);
    free(mpi_offset);
    // Make false_far be in ascending order
    qsort(false_far, nblocks_false_far, sizeof(*false_far), cmp_size_t);
    if(nblocks_false_far > 0)
    {
        // Update list of near-field blocks
        new_nblocks_near = nblocks_near+nblocks_false_far;
        new_nblocks_near_local = nblocks_near_local+nblocks_false_far_local;
        STARSH_MALLOC(block_near, 2*new_nblocks_near);
        if(new_nblocks_near_local > 0)
            STARSH_MALLOC(block_near_local, new_nblocks_near_local);
        // At first get all near-field blocks, assumed to be dense
        #pragma omp parallel for schedule(static)
        for(bi = 0; bi < 2*nblocks_near; bi++)
            block_near[bi] = F->block_near[bi];
        #pragma omp parallel for schedule(static)
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
            block_near_local[lbi] = F->block_near_local[lbi];
        // Add false far-field blocks
        #pragma omp parallel for schedule(static)
        for(bi = 0; bi < nblocks_false_far; bi++)
        {
            STARSH_int bj = false_far[bi];
            block_near[2*(bi+nblocks_near)] = F->block_far[2*bj];
            block_near[2*(bi+nblocks_near)+1] = F->block_far[2*bj+1];
        }
        bi = 0;
        for(lbi = 0; lbi < nblocks_false_far_local; lbi++)
        {
            lbj = false_far_local[lbi];
            while(false_far[bi] < lbj)
                bi++;
            block_near_local[nblocks_near_local+lbi] = nblocks_near+bi;
        }
        // Update list of far-field blocks
        new_nblocks_far = nblocks_far-nblocks_false_far;
        new_nblocks_far_local = nblocks_far_local-nblocks_false_far_local;
        if(new_nblocks_far > 0)
        {
            STARSH_MALLOC(block_far, 2*new_nblocks_far);
            if(new_nblocks_far_local > 0)
                STARSH_MALLOC(block_far_local, new_nblocks_far_local);
            bj = 0;
            lbi = 0;
            lbj = 0;
            for(bi = 0; bi < nblocks_far; bi++)
            {
                // `false_far` must be in ascending order for this to work
                if(false_far[bj] == bi)
                {
                    if(nblocks_false_far_local > lbj &&
                            false_far_local[lbj] == bi)
                    {
                        lbi++;
                        lbj++;
                    }
                    bj++;
                }
                else
                {
                    block_far[2*(bi-bj)] = F->block_far[2*bi];
                    block_far[2*(bi-bj)+1] = F->block_far[2*bi+1];
                    if(nblocks_far_local > lbi &&
                            F->block_far_local[lbi] == bi)
                    {
                        block_far_local[lbi-lbj] = bi-bj;
                        lbi++;
                    }
                }
            }
        }
        // Update format by creating new format
        STARSH_blrf *F2;
        info = starsh_blrf_new_from_coo_mpi(&F2, P, F->symm, RC, CC,
                new_nblocks_far, block_far, new_nblocks_far_local,
                block_far_local, new_nblocks_near, block_near,
                new_nblocks_near_local, block_near_local, F->type);
        // Swap internal data of formats and free unnecessary data
        STARSH_blrf tmp_blrf = *F;
        *F = *F2;
        *F2 = tmp_blrf;
        if(mpi_rank == 0)
            STARSH_WARNING("`F` was modified due to false far-field blocks");
        starsh_blrf_free(F2);
    }
    // Compute near-field blocks if needed
    if(onfly == 0 && new_nblocks_near > 0)
    {
        STARSH_MALLOC(near_D, new_nblocks_near_local);
        size_t size_D = 0;
        // Simple cycle over all near-field blocks
        for(lbi = 0; lbi < new_nblocks_near_local; lbi++)
        {
            STARSH_int bi = block_near_local[lbi];
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
        #pragma omp parallel for schedule(dynamic, 1)
        for(lbi = 0; lbi < new_nblocks_near_local; lbi++)
        {
            STARSH_int bi = block_near_local[lbi];
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
                offset_D += nrows*ncols;
                //array_from_buffer(near_D+lbi, 2, shape, 'd', 'F', D);
                //offset_D += near_D[lbi]->size;
            }
            array_from_buffer(near_D+lbi, 2, shape, 'd', 'F', D);
#ifdef OPENMP
            double time0 = omp_get_wtime();
#endif
            kernel(nrows, ncols, RC->pivot+RC->start[i],
                    CC->pivot+CC->start[j], RD, CD, D, nrows);
#ifdef OPENMP
            double time1 = omp_get_wtime();
            #pragma omp critical
            kernel_time += time1-time0;
#endif
        }
    }
    // Change sizes of far_rank, far_U and far_V if there were false
    // far-field blocks
    lbj = 0;
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
    {
        if(far_rank[lbi] == -1)
            lbj++;
        else
        {
            int shape_U[2] = {far_U[lbi]->shape[0], far_rank[lbi]};
            int shape_V[2] = {far_V[lbi]->shape[0], far_rank[lbi]};
            array_from_buffer(far_U+lbi-lbj, 2, shape_U, 'd', 'F',
                    far_U[lbi]->data);
            array_from_buffer(far_V+lbi-lbj, 2, shape_V, 'd', 'F',
                    far_V[lbi]->data);
            far_rank[lbi-lbj] = far_rank[lbi];
        }
    }
    if(nblocks_false_far_local > 0 && new_nblocks_far_local > 0)
    {
        STARSH_REALLOC(far_rank, new_nblocks_far_local);
        STARSH_REALLOC(far_U, new_nblocks_far_local);
        STARSH_REALLOC(far_V, new_nblocks_far_local);
    }
    // If all far-field blocks are false, then dealloc buffers
    if(new_nblocks_far_local == 0 && nblocks_far_local > 0)
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
    if(nblocks_false_far_local > 0)
        free(false_far_local);
    // Finish with creating instance of Block Low-Rank Matrix with given
    // buffers
#ifdef OPENMP
    double mpi_drsdd_time = 0, mpi_kernel_time = 0;
    MPI_Reduce(&drsdd_time, &mpi_drsdd_time, 1, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);
    MPI_Reduce(&kernel_time, &mpi_kernel_time, 1, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);
    if(mpi_rank == 0)
    {
        //STARSH_WARNING("DRSDD kernel total time: %e secs", mpi_drsdd_time);
        //STARSH_WARNING("MATRIX kernel total time: %e secs", mpi_kernel_time);
    }
#endif
    return starsh_blrm_new_mpi(matrix, F, far_rank, far_U, far_V, onfly,
            near_D, alloc_U, alloc_V, alloc_D, '1');
}
