/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/blrm/dqp3.c
 * @version 0.3.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-starpu.h"

int starsh_blrm__dqp3_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly)
//! Approximate each tile of BLR matrix with RRQR (GEQP3 function).
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
    const int oversample = starsh_params.oversample;
    struct starpu_codelet codelet =
    {
        .cpu_funcs = {starsh_dense_dlrqp3_starpu},
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
    STARSH_int bi_value[nblocks_far];
    starpu_data_handle_t bi_handle[nblocks_far], rank_handle[nblocks_far];
    starpu_data_handle_t U_handle[nblocks_far], V_handle[nblocks_far];
    starpu_data_handle_t work_handle[nblocks_far], iwork_handle[nblocks_far];
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
            int nrows = RC->size[i], ncols = CC->size[j];
            int mn = nrows < ncols ? nrows : ncols;
            int mn2 = maxrank+oversample;
            if(mn2 > mn)
                mn2 = mn;
            // Get size of temporary arrays
            int lwork = 3*ncols+1, lwork_sdd = (4*mn2+7)*mn2;
            if(lwork_sdd > lwork)
                lwork = lwork_sdd;
            lwork += nrows*ncols+mn2*(2*ncols+mn2+1)+mn;
            int liwork = ncols, liwork_sdd = 8*mn2;
            if(liwork_sdd > liwork)
                liwork = liwork_sdd;
            int shape_U[] = {nrows, maxrank};
            int shape_V[] = {ncols, maxrank};
            double *U = alloc_U+offset_U, *V = alloc_V+offset_V;
            offset_U += nrows*maxrank;
            offset_V += ncols*maxrank;
            array_from_buffer(far_U+bi, 2, shape_U, 'd', 'F', U);
            array_from_buffer(far_V+bi, 2, shape_V, 'd', 'F', V);
            bi_value[bi] = bi;
            starpu_variable_data_register(bi_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(bi_value+bi), sizeof(*bi_value));
            starpu_variable_data_register(rank_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(far_rank+bi), sizeof(*far_rank));
            starpu_vector_data_register(U_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(far_U[bi]->data), nrows*maxrank, sizeof(*U));
            starpu_vector_data_register(V_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(far_V[bi]->data), ncols*maxrank, sizeof(*V));
            starpu_vector_data_register(work_handle+bi, -1, 0, lwork,
                    sizeof(*U));
            starpu_vector_data_register(iwork_handle+bi, -1, 0, liwork,
                    sizeof(int));
        }
        offset_U = 0;
        offset_V = 0;
    }
    // Work variables
    int info;
    // Simple cycle over all far-field admissible blocks
    for(bi = 0; bi < nblocks_far; bi++)
    {
        starpu_task_insert(&codelet, STARPU_VALUE, &F, sizeof(F),
                STARPU_VALUE, &maxrank, sizeof(maxrank),
                STARPU_VALUE, &oversample, sizeof(oversample),
                STARPU_VALUE, &tol, sizeof(tol),
                STARPU_R, bi_handle[bi], STARPU_W, rank_handle[bi],
                STARPU_W, U_handle[bi], STARPU_W, V_handle[bi],
                STARPU_SCRATCH, work_handle[bi],
                STARPU_SCRATCH, iwork_handle[bi],
                0);
        starpu_data_unregister_submit(bi_handle[bi]);
        starpu_data_unregister_submit(rank_handle[bi]);
        starpu_data_unregister_submit(U_handle[bi]);
        starpu_data_unregister_submit(V_handle[bi]);
        starpu_data_unregister_submit(work_handle[bi]);
        starpu_data_unregister_submit(iwork_handle[bi]);
    }
    starpu_task_wait_for_all();
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
        for(bi = 0; bi < 2*nblocks_near; bi++)
            block_near[bi] = F->block_near[bi];
        // Add false far-field blocks
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
        STARSH_int nbi_value[new_nblocks_near];
        starpu_data_handle_t D_handle[new_nblocks_near];
        starpu_data_handle_t nbi_handle[new_nblocks_near];
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
        for(bi = 0; bi < new_nblocks_near; bi++)
        {
            // Get indexes of corresponding block row and block column
            STARSH_int i = block_near[2*bi];
            STARSH_int j = block_near[2*bi+1];
            // Get corresponding sizes and minimum of them
            int nrows = RC->size[i];
            int ncols = CC->size[j];
            int shape[2] = {nrows, ncols};
            double *D = alloc_D+offset_D;
            array_from_buffer(near_D+bi, 2, shape, 'd', 'F', D);
            offset_D += near_D[bi]->size;
            nbi_value[bi] = bi;
            starpu_variable_data_register(nbi_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(nbi_value+bi), sizeof(*nbi_value));
            starpu_vector_data_register(D_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(near_D[bi]->data), (size_t)nrows*(size_t)ncols,
                    sizeof(*D));
        }
        for(bi = 0; bi < new_nblocks_near; bi++)
        {
            starpu_task_insert(&codelet2, STARPU_VALUE, &F, sizeof(F),
                    STARPU_R, nbi_handle[bi], STARPU_W, D_handle[bi],
                    0);
            starpu_data_unregister_submit(nbi_handle[bi]);
            starpu_data_unregister_submit(D_handle[bi]);
        }
        // Wait in this scope, because all handles are not visible outside
        starpu_task_wait_for_all();
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
    return starsh_blrm_new(matrix, F, far_rank, far_U, far_V, onfly, near_D,
            alloc_U, alloc_V, alloc_D, '1');
}

