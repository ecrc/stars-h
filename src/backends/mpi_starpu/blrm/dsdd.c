/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/mpi_starpu/blrm/dsdd.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-mpi-starpu.h"

int starsh_blrm__dsdd_mpi_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly)
//! Approximate each tile by divide-and-conquer SVD (GESDD function).
/*!
 * @param[out] matrix: Address of pointer to @ref STARSH_blrm object.
 * @param[in] format: Block low-rank format.
 * @param[in] maxrank: Maximum possible rank.
 * @param[in] tol: Relative error tolerance.
 * @param[in] onfly: Whether not to store dense blocks.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup blrm
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
    struct starpu_codelet codelet =
    {
        .cpu_funcs = {starsh_dense_dlrsdd_starpu},
        .nbuffers = 6,
        .modes = {STARPU_R, STARPU_W, STARPU_W, STARPU_W, STARPU_SCRATCH,
            STARPU_SCRATCH}
    };
    struct starpu_codelet codelet2 =
    {
        .cpu_funcs = {starsh_dense_kernel_starpu},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W}
    };
    STARSH_int bi_value[nblocks_far_local];
    starpu_data_handle_t bi_handle[nblocks_far_local];
    starpu_data_handle_t rank_handle[nblocks_far_local];
    starpu_data_handle_t U_handle[nblocks_far_local];
    starpu_data_handle_t V_handle[nblocks_far_local];
    starpu_data_handle_t work_handle[nblocks_far_local];
    starpu_data_handle_t iwork_handle[nblocks_far_local];
    // Init buffers to store low-rank factors of far-field blocks if needed
    if(nblocks_far_local > 0)
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
            int mn = nrows < ncols ? nrows : ncols;
            // Get size of temporary arrays
            int lmn = mn, lwork = (4*lmn+8+nrows+ncols)*lmn, liwork = 8*lmn;
            double *U = alloc_U+offset_U, *V = alloc_V+offset_V;
            offset_U += nrows*maxrank;
            offset_V += ncols*maxrank;
            array_from_buffer(far_U+lbi, 2, shape_U, 'd', 'F', U);
            array_from_buffer(far_V+lbi, 2, shape_V, 'd', 'F', V);
            bi_value[lbi] = bi;
            starpu_variable_data_register(bi_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(bi_value+lbi), sizeof(*bi_value));
            starpu_variable_data_register(rank_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(far_rank+lbi), sizeof(*far_rank));
            starpu_vector_data_register(U_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(far_U[lbi]->data), nrows*maxrank, sizeof(*U));
            starpu_vector_data_register(V_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(far_V[lbi]->data), ncols*maxrank, sizeof(*V));
            starpu_vector_data_register(work_handle+lbi, -1, 0, lwork,
                    sizeof(*U));
            starpu_vector_data_register(iwork_handle+lbi, -1, 0, liwork,
                    sizeof(int));
        }
        offset_U = 0;
        offset_V = 0;
    }
    // Work variables
    int info;
    // Simple cycle over all far-field admissible blocks
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
    {
        starpu_task_insert(&codelet, STARPU_VALUE, &F, sizeof(F),
                STARPU_VALUE, &maxrank, sizeof(maxrank),
                STARPU_VALUE, &tol, sizeof(tol),
                STARPU_R, bi_handle[lbi], STARPU_W, rank_handle[lbi],
                STARPU_W, U_handle[lbi], STARPU_W, V_handle[lbi],
                STARPU_SCRATCH, work_handle[lbi],
                STARPU_SCRATCH, iwork_handle[lbi],
                0);
        starpu_data_unregister_submit(bi_handle[lbi]);
        starpu_data_unregister_submit(rank_handle[lbi]);
        starpu_data_unregister_submit(U_handle[lbi]);
        starpu_data_unregister_submit(V_handle[lbi]);
        starpu_data_unregister_submit(work_handle[lbi]);
        starpu_data_unregister_submit(iwork_handle[lbi]);
    }
    starpu_task_wait_for_all();
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
    int int_nblocks_false_far_local = nblocks_false_far_local;
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
    STARSH_int *false_far = NULL;
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
        for(bi = 0; bi < 2*nblocks_near; bi++)
            block_near[bi] = F->block_near[bi];
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
            block_near_local[lbi] = F->block_near_local[lbi];
        // Add false far-field blocks
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
            while(bi < nblocks_false_far && false_far[bi] < lbj)
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
                if(bj < nblocks_false_far && false_far[bj] == bi)
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
    if(onfly == 0 && new_nblocks_near_local > 0)
    {
        STARSH_int nbi_value[new_nblocks_near_local];
        starpu_data_handle_t D_handle[new_nblocks_near_local];
        starpu_data_handle_t nbi_handle[new_nblocks_near_local];
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
            D = alloc_D+offset_D;
            offset_D += nrows*ncols;
            array_from_buffer(near_D+lbi, 2, shape, 'd', 'F', D);
            nbi_value[lbi] = bi;
            starpu_variable_data_register(nbi_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(nbi_value+lbi), sizeof(*nbi_value));
            starpu_vector_data_register(D_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(near_D[lbi]->data),
                    (size_t)nrows*(size_t)ncols, sizeof(*D));
        }
        for(lbi = 0; lbi < new_nblocks_near_local; lbi++)
        {
            starpu_task_insert(&codelet2, STARPU_VALUE, &F, sizeof(F),
                    STARPU_R, nbi_handle[lbi], STARPU_W, D_handle[lbi],
                    0);
            starpu_data_unregister_submit(nbi_handle[lbi]);
            starpu_data_unregister_submit(D_handle[lbi]);
        }
        // Wait in this scope, because all handles are not visible outside
        starpu_task_wait_for_all();
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
    return starsh_blrm_new_mpi(matrix, F, far_rank, far_U, far_V, onfly,
            near_D, alloc_U, alloc_V, alloc_D, '1');
}

