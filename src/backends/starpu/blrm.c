#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>
#include <mkl.h>
#include "stars.h"
#include "misc.h"

struct _params_t
{
    STARS_BLRF *F;
    int fixrank;
    double tol;
    int onfly;
    int maxrank;
};

static void tiled_compress_algebraic_svd_kernel(void *buffer[], void *cl_arg)
{
    struct _params_t *params= cl_arg;
    STARS_BLRF *F = params->F;
    int fixrank = params->fixrank;
    double tol = params->tol;
    int onfly = params->onfly;
    int maxrank = params->maxrank;
    STARS_Problem *P = F->problem;
    // Shortcuts to information about tiles
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    int rank;
    size_t bi = *(size_t *)STARPU_VARIABLE_GET_PTR(buffer[0]);
    int i = F->block_far[2*bi];
    int j = F->block_far[2*bi+1];
    size_t nrowsi = R->size[i];
    size_t ncolsj = C->size[j];
    int ndim = 2;
    int shape[ndim];
    shape[0] = nrowsi;
    shape[ndim-1] = ncolsj;
    size_t mn = nrowsi > ncolsj ? ncolsj : nrowsi;
    double *tmp_ptr = STARPU_VARIABLE_GET_PTR(buffer[1]);
    double *A = tmp_ptr, *UV = tmp_ptr+nrowsi*ncolsj, *S = UV+mn*mn;
    double *work = S+mn;
    double *U = A, *V = UV;
    if(nrowsi < ncolsj)
    {
        U = UV;
        V = A;
    }
    size_t tmp_size2 = (5*mn+7)*mn, tmp_size3 = nrowsi+ncolsj+2*mn;
    size_t tmp_size = tmp_size2 > tmp_size3 ? tmp_size2 : tmp_size3;
    int *iwork = (int *)(work+tmp_size);
    int *far_rank = STARPU_VARIABLE_GET_PTR(buffer[2]);
    double *far_U = STARPU_VARIABLE_GET_PTR(buffer[3]);
    double *far_V = STARPU_VARIABLE_GET_PTR(buffer[4]);
    // Get array, holding necessary data
    P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
            C->pivot+C->start[j], P->row_data, P->col_data, A);
    LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'O', nrowsi, ncolsj, A, nrowsi, S, U,
            nrowsi, V, mn, work, tmp_size, iwork);
    // Get rank
    double Stol = 0, Stmp = S[mn-1]*S[mn-1];
    rank = mn;
    for(size_t i = 0; i < mn; i++)
        Stol += S[i]*S[i];
    // If total norm is 0, then rank is 0, otherwise rank is > 0
    if(Stol == 0)
        rank = 0;
    Stol *= tol*tol;
    while(rank > 1 && Stol >= Stmp)
    {
        rank--;
        Stmp += S[rank-1]*S[rank-1];
    }
    //printf("nrows=%d ncols=%d mn=%d rank=%d Stol=%e\n", nrowsi, ncolsj, mn, rank, Stol);
    if(rank < mn/2)
    // If block is low-rank
    {
        if(fixrank != 0)
            rank = fixrank;
        far_rank[0] = rank;
        shape[0] = nrowsi;
        shape[1] = rank;
        shape[0] = rank;
        shape[1] = ncolsj;
        // Copy part of `U` into low-rank factor `far_U`
        cblas_dcopy(rank*nrowsi, U, 1, far_U, 1);
        //ptr = far_V[bi]->data;
        // Copy part of `S`*`V` into low-rank factor `far_V`
        for(size_t k = 0; k < ncolsj; k++)
            for(size_t l = 0; l < rank; l++)
            {
                far_V[k*rank+l] = S[l]*V[k*mn+l];
            }
    }
    else
    // If block is NOT low-rank
    {
        far_rank[0] = -1;
    }
}

int STARS_BLRM_tiled_compress_algebraic_svd_starpu(STARS_BLRM **M,
        STARS_BLRF *F, int fixrank, double tol, int onfly, int maxrank)
// Static scheduling of work on different MPI nodes (for simplicity) plus
// dynamic scheduling of work on a single MPI node by means of StarPU
{
    //Set timer
    struct timeval tmp_time, tmp_time2;
    gettimeofday(&tmp_time, NULL);
    // Check parameters
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(F == NULL)
    {
        STARS_ERROR("invalid value of `F`");
        return 1;
    }
    if(fixrank < 0)
    {
        STARS_ERROR("invalid value of `fixrank`");
        return 1;
    }
    if(tol < 0 || tol >= 1)
    {
        STARS_ERROR("invalid value of `tol`");
        return 1;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARS_ERROR("invalid value of `onfly`");
        return 1;
    }
    STARS_Problem *P = F->problem;
    int ndim = P->ndim;
    if(ndim != 2)
    {
        STARS_ERROR("only scalar kernels are supported");
        return 1;
    }
    // Shortcuts to information about tiles
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    STARS_BLRF *F2;
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near;
    // Following values default to given block low-rank format F, but they are
    // changed when there are false far-field blocks.
    size_t new_nblocks_far = nblocks_far, new_nblocks_near = nblocks_near;
    int *block_far = F->block_far, *block_near = F->block_near;
    // Places to store low-rank factors, dense blocks and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank = NULL;
    // Allocate memory for far-field blocks
    if(nblocks_far > 0)
    {
        STARS_MALLOC(far_U, nblocks_far);
        STARS_MALLOC(far_V, nblocks_far);
        STARS_MALLOC(far_rank, nblocks_far);
    }
    // Work variables
    int nrowsi, ncolsj;
    int shape[ndim];
    size_t bi, bj, k, l;
    struct starpu_codelet codelet =
    {
        .cpu_funcs = {tiled_compress_algebraic_svd_kernel},
        .nbuffers = 5,
        .modes = {STARPU_R, STARPU_SCRATCH, STARPU_W, STARPU_W, STARPU_W}
    };
    starpu_init(NULL);
    size_t bi_value[nblocks_far];
    starpu_data_handle_t bi_handle[nblocks_far], far_rank_handle[nblocks_far];
    starpu_data_handle_t tmp_handle[nblocks_far], U_handle[nblocks_far];
    starpu_data_handle_t V_handle[nblocks_far];
    struct _params_t params = {F, fixrank, tol, onfly, maxrank};
    for(bi = 0; bi < nblocks_far; bi++)
    {
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        size_t nrowsi = R->size[i];
        size_t ncolsj = C->size[j];
        size_t mn = nrowsi > ncolsj ? ncolsj : nrowsi;
        struct starpu_task *task = starpu_task_create();
        task->cl = &codelet;
        bi_value[bi] = bi;
        starpu_variable_data_register(&bi_handle[bi], STARPU_MAIN_RAM,
                (uintptr_t)&bi_value[bi], sizeof(*bi_value));
        size_t tmp_size2 = (5*mn+7)*mn, tmp_size3 = nrowsi+ncolsj+2*mn;
        size_t tmp_size = tmp_size2 > tmp_size3 ? tmp_size2 : tmp_size3;
        tmp_size += (nrowsi+ncolsj)*mn+mn;
        tmp_size = tmp_size*sizeof(double)+8*mn*sizeof(int);
        starpu_variable_data_register(&tmp_handle[bi], -1, (uintptr_t)NULL,
                tmp_size);
        starpu_variable_data_register(&far_rank_handle[bi], STARPU_MAIN_RAM,
                (uintptr_t)&far_rank[bi], sizeof(*far_rank));
        int shape[2] = {nrowsi, maxrank};
        Array_new(far_U+bi, 2, shape, 'd', 'F');
        shape[0] = maxrank, shape[1] = ncolsj;
        Array_new(far_V+bi, 2, shape, 'd', 'F');
        starpu_variable_data_register(&U_handle[bi], STARPU_MAIN_RAM,
                (uintptr_t)far_U[bi]->data, far_U[bi]->data_nbytes);
        starpu_variable_data_register(&V_handle[bi], STARPU_MAIN_RAM,
                (uintptr_t)far_V[bi]->data, far_V[bi]->data_nbytes);
        task->handles[0] = bi_handle[bi];
        task->handles[1] = tmp_handle[bi];
        task->handles[2] = far_rank_handle[bi];
        task->handles[3] = U_handle[bi];
        task->handles[4] = V_handle[bi];
        task->cl_arg = &params;
        task->cl_arg_size = sizeof(params);
        starpu_task_submit(task);
        starpu_data_unregister_submit(bi_handle[bi]);
        starpu_data_unregister_submit(tmp_handle[bi]);
        starpu_data_unregister_submit(far_rank_handle[bi]);
        starpu_data_unregister_submit(U_handle[bi]);
        starpu_data_unregister_submit(V_handle[bi]);
    }
    starpu_task_wait_for_all();
    starpu_shutdown();
    for(bi = 0; bi < nblocks_far; bi++)
    {
        if(far_rank[bi] >= 0)
        {
            //far_U[bi]->shape[1] = far_rank[bi];
            //far_V[bi]->shape[0] = far_rank[bi];
            int shape[2] = {far_U[bi]->shape[0], far_rank[bi]};
            Array *tmp = far_U[bi];
            Array_new(far_U+bi, 2, shape, 'd', 'F');
            cblas_dcopy(shape[0]*shape[1], tmp->data, 1, far_U[bi]->data, 1);
            Array_free(tmp);
            shape[0] = far_rank[bi], shape[1] = far_V[bi]->shape[1];
            tmp = far_V[bi];
            Array_new(far_V+bi, 2, shape, 'd', 'F');
            cblas_dcopy(shape[0]*shape[1], tmp->data, 1, far_V[bi]->data, 1);
            Array_free(tmp);
        }
        //printf("rank %d\n", far_rank[bi]);
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
        STARS_MALLOC(false_far, nblocks_false_far);
        size_t bj = 0;
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
        STARS_MALLOC(block_near, 2*new_nblocks_near);
        for(bi = 0; bi < 2*nblocks_near; bi++)
            block_near[bi] = F->block_near[bi];
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
            STARS_MALLOC(block_far, 2*new_nblocks_far);
            size_t bj = 0;
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
        // Update format be creating new format and swapping buffers with it
        STARS_BLRF *F2;
        int info = STARS_BLRF_new(&F2, P, F->symm, R, C, new_nblocks_far,
                block_far, new_nblocks_near, block_near, F->type);
        // Swap internal data of formats and free unnecessary data
        STARS_BLRF_swap(F, F2);
        STARS_WARNING("`F` was modified due to false far-field blocks");
        info = STARS_BLRF_free(F2);
    }
    // Changing size of far_rank, far_U and far_V
    if(nblocks_false_far > 0 && new_nblocks_far > 0)
    {
        size_t bj = 0;
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
        STARS_REALLOC(far_rank, new_nblocks_far);
        STARS_REALLOC(far_U, new_nblocks_far);
        STARS_REALLOC(far_V, new_nblocks_far);
    }
    if(new_nblocks_far == 0 && nblocks_far > 0)
    // If all far-field blocks are false, then dealloc buffers
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
    void *alloc_U = NULL, *alloc_V = NULL, *alloc_D = NULL;
    gettimeofday(&tmp_time2, NULL);
    double time = tmp_time2.tv_sec-tmp_time.tv_sec+
            (tmp_time2.tv_usec-tmp_time.tv_usec)*1e-6;
    STARS_WARNING("total time: %f sec", time);
    return STARS_BLRM_new(M, F, far_rank, far_U, far_V, onfly, near_D, alloc_U,
            alloc_V, alloc_D, '2');
}
