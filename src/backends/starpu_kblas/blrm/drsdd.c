/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/blrm/drsdd.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-starpu-kblas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <kblas.h>
#include "batch_rand.h"
#include <starpu.h>
#include <omp.h>

static void init_starpu_kblas(void *args)
{
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    kblasRandState_t *kblas_states;
    cudaStream_t stream = starpu_cuda_get_local_stream();
    int nb, nsamples, maxbatch;
    double **work;
    int **iwork;
    starpu_codelet_unpack_args(args, &cublas_handles, &kblas_handles,
            &kblas_states, &work, &iwork, &nb, &nsamples, &maxbatch);
    int id = starpu_worker_get_id();
    cublasStatus_t status;
    kblasCreate(&kblas_handles[id]);
    kblasSetStream(kblas_handles[id], stream);
    kblasDrsvd_batch_wsquery(kblas_handles[id], nb, nb, nsamples, maxbatch);
    kblasAllocateWorkspace(kblas_handles[id]);
    cublas_handles[id] = kblasGetCublasHandle(kblas_handles[id]);
    kblasInitRandState(kblas_handles[id], &kblas_states[id], 16384*2, 0);
    work[id] = malloc(nsamples*maxbatch*sizeof(double));
    iwork[id] = malloc(maxbatch*sizeof(int));
    cudaStreamSynchronize(stream);
}

static void init_starpu_cpu(void *args)
{
    int nb, nsamples;
    int lwork, liwork;
    double **work;
    int **iwork;
    starpu_codelet_unpack_args(args, &nb, &nsamples, &work, &lwork, &iwork,
            &liwork);
    int id = starpu_worker_get_id();
    work[id] = malloc(lwork*sizeof(*work[0]));
    iwork[id] = malloc(liwork*sizeof(*iwork[0]));
}

static void deinit_starpu_kblas(void *args)
{
    int nb, nsamples, maxbatch;
    double **work;
    int **iwork;
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    kblasRandState_t *kblas_states;
    starpu_codelet_unpack_args(args, &cublas_handles, &kblas_handles,
            &kblas_states, &work, &iwork, &nb, &nsamples, &maxbatch);
    int id = starpu_worker_get_id();
    kblasDestroyRandState(kblas_states[id]);
    kblasDestroy(&kblas_handles[id]);
    free(work[id]);
    free(iwork[id]);
}

static void deinit_starpu_cpu(void *args)
{
    int nb, nsamples;
    int lwork, liwork;
    double **work;
    int **iwork;
    starpu_codelet_unpack_args(args, &nb, &nsamples, &work, &lwork, &iwork,
            &liwork);
    int id = starpu_worker_get_id();
    free(work[id]);
    free(iwork[id]);
}

static void empty_cpu_func(void *buffer[],  void *cl_arg)
{
}

int starsh_blrm__drsdd_starpu_kblas(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly)
//! Approximate each tile by randomized SVD.
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
    STARSH_int bi, bj = 0;
    const int oversample = starsh_params.oversample;
    // Init CuBLAS and KBLAS handles and temp buffers for all workers (but they
    // are used only in GPU codelets)
    int workers = starpu_worker_get_count();
    cublasHandle_t cublas_handles[workers];
    kblasHandle_t kblas_handles[workers];
    kblasRandState_t kblas_states[workers];
    double *work[workers];
    int *iwork[workers];
    cublasHandle_t *cuhandles = cublas_handles;
    kblasHandle_t *khandles = kblas_handles;
    kblasRandState_t *kstates = kblas_states;
    double **wwork = work;
    int **wiwork = iwork;
    //printf("MAIN: %p, %p, %p\n", cuhandles, khandles, svhandles);
    void *args_gpu, *args_cpu;
    size_t args_gpu_size = 0;
    size_t args_cpu_size = 0;
    // This works only for TLR with equal tiles
    int nb = RC->size[0];
    int nsamples = maxrank+oversample;
    // Set size of batch
    int batch_size = 100;
    // Ceil number of batches
    int nbatches = (nblocks_far-1)/batch_size + 1;
    // Get corresponding sizes and minimum of them
    int mn = maxrank+oversample;
    if(mn > nb)
        mn = nb;
    // Get size of temporary arrays
    int lwork = nb;
    int lwork_sdd = (4*mn+7) * mn;
    if(lwork_sdd > lwork)
        lwork = lwork_sdd;
    lwork += mn*(3*nb+mn+1) + nb*nb;
    int liwork = 8 * mn;
    starpu_codelet_pack_args(&args_gpu, &args_gpu_size,
            STARPU_VALUE, &cuhandles, sizeof(cuhandles),
            STARPU_VALUE, &khandles, sizeof(khandles),
            STARPU_VALUE, &kstates, sizeof(kstates),
            STARPU_VALUE, &wwork, sizeof(wwork),
            STARPU_VALUE, &wiwork, sizeof(wiwork),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &nsamples, sizeof(nsamples),
            STARPU_VALUE, &batch_size, sizeof(batch_size),
            0);
    starpu_codelet_pack_args(&args_cpu, &args_cpu_size,
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &nsamples, sizeof(nsamples),
            STARPU_VALUE, &wwork, sizeof(wwork),
            STARPU_VALUE, &lwork, sizeof(lwork),
            STARPU_VALUE, &wiwork, sizeof(wiwork),
            STARPU_VALUE, &liwork, sizeof(liwork),
            0);
    starpu_execute_on_each_worker(init_starpu_kblas, args_gpu, STARPU_CUDA);
    starpu_execute_on_each_worker(init_starpu_cpu, args_cpu, STARPU_CPU);
    // Init codelet structs and handles
    struct starpu_codelet codelet_kernel =
    {
        .cpu_funcs = {starsh_dense_kernel_starpu_kblas_cpu},
        .nbuffers = 2,
        .modes = {STARPU_W, STARPU_R},
        .type = STARPU_SPMD,
        .max_parallelism = INT_MAX,
    };
    struct starpu_codelet codelet_lowrank =
    {
        .cpu_funcs = {starsh_dense_dlrrsdd_starpu_kblas_cpu},
        .cuda_funcs = {starsh_dense_dlrrsdd_starpu_kblas_gpu},
        .cuda_flags = {STARPU_CUDA_ASYNC},
        .nbuffers = 5,
        .modes = {STARPU_R, STARPU_SCRATCH, STARPU_W, STARPU_W, STARPU_W},
        .type = STARPU_SPMD,
        .max_parallelism = INT_MAX,
    };
    struct starpu_codelet codelet_lowrank_cpu =
    {
        .cpu_funcs = {starsh_dense_dlrrsdd_starpu_kblas_cpu},
        .nbuffers = 5,
        .modes = {STARPU_R, STARPU_SCRATCH, STARPU_W, STARPU_W, STARPU_W},
        .type = STARPU_SPMD,
        .max_parallelism = INT_MAX,
    };
    struct starpu_codelet codelet_lowrank_gpu =
    {
        .cuda_funcs = {starsh_dense_dlrrsdd_starpu_kblas_gpu},
        .cuda_flags = {STARPU_CUDA_ASYNC},
        .nbuffers = 5,
        .modes = {STARPU_R, STARPU_SCRATCH, STARPU_W, STARPU_W, STARPU_W},
    };
    struct starpu_codelet codelet_get_data_back_to_cpu =
    {
        .cpu_funcs = {empty_cpu_func},
        .nbuffers = 1,
        .modes = {STARPU_R},
    };
    // Select if ONLY cpu or gpu
    if(getenv("STARSH_KBLAS_CPU"))
        codelet_lowrank = codelet_lowrank_cpu;
    else if(getenv("STARSH_KBLAS_GPU"))
        codelet_lowrank = codelet_lowrank_gpu;
    starpu_data_handle_t rank_handle[nbatches];
    starpu_data_handle_t D_handle[nbatches];
    starpu_data_handle_t Dcopy_handle[nbatches];
    starpu_data_handle_t index_handle[nbatches];
    starpu_data_handle_t U_handle[nbatches];
    starpu_data_handle_t V_handle[nbatches];
    //printf("BATCHSIZE=%d BATCHCOUNT=%d\n", batch_size, nbatches);
    // Init buffers to store low-rank factors of far-field blocks if needed
    if(nbatches > 0)
    {
        STARSH_MALLOC(far_U, nblocks_far);
        STARSH_MALLOC(far_V, nblocks_far);
        STARSH_MALLOC(far_rank, nblocks_far);
        size_t size_U = nblocks_far * nb * maxrank;
        size_t size_V = size_U;
        STARSH_MALLOC(alloc_U, size_U);
        STARSH_MALLOC(alloc_V, size_V);
        int shape[] = {nb, maxrank};
        for(bi = 0; bi < nblocks_far; ++bi)
        {
            STARSH_int offset = bi * nb * maxrank;
            array_from_buffer(far_U+bi, 2, shape, 'd', 'F', alloc_U+offset);
            array_from_buffer(far_V+bi, 2, shape, 'd', 'F', alloc_V+offset);
        }
        // START MEASURING TIME
        double time0 = omp_get_wtime();
        for(bi = 0; bi < nbatches; ++bi)
        {
            STARSH_int offset = bi * batch_size * nb * maxrank;
            double *U = alloc_U + offset;
            double *V = alloc_V + offset;
            STARSH_int this_batch_size = nblocks_far - bi*batch_size;
            if(this_batch_size > batch_size)
                this_batch_size = batch_size;
            STARSH_int D_size = this_batch_size * nb * nb;
            STARSH_int U_size = this_batch_size * nb * maxrank;
            STARSH_int V_size = U_size;
            //printf("THIS BATCH SIZE=%d\n", this_batch_size);
            starpu_vector_data_register(rank_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(far_rank + bi*batch_size), this_batch_size,
                    sizeof(*far_rank));
            starpu_vector_data_register(D_handle+bi, -1, 0, D_size,
                    sizeof(double));
            starpu_vector_data_register(Dcopy_handle+bi, -1, 0, D_size,
                    sizeof(double));
            starpu_vector_data_register(index_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(block_far + 2*bi*batch_size),
                    2*this_batch_size, sizeof(*block_far));
            starpu_vector_data_register(U_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(U), U_size, sizeof(*U));
            starpu_vector_data_register(V_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(V), V_size, sizeof(*V));
        }
        //printf("REGISTER DATA IN: %f seconds\n", omp_get_wtime()-time0);
    }
    // Work variables
    int info;
    // START MEASURING TIME
    double time0 = omp_get_wtime();
    for(bi = 0; bi < nbatches; ++bi)
    {
        //printf("RUNNING BATCH=%d\n", bi);
        STARSH_int this_batch_size = nblocks_far - bi*batch_size;
        if(this_batch_size > batch_size)
            this_batch_size = batch_size;
        // Generate matrix
        starpu_task_insert(&codelet_kernel,
                STARPU_VALUE, &F, sizeof(F),
                STARPU_VALUE, &this_batch_size, sizeof(this_batch_size),
                STARPU_W, D_handle[bi],
                STARPU_R, index_handle[bi],
                0);
        starpu_data_unregister_submit(index_handle[bi]);
    }
    //starpu_task_wait_for_all();
    //double time1 = omp_get_wtime();
    //printf("COMPUTE MATRIX IN: %f seconds\n", time1-time0);
    //time0 = time1;
    STARSH_int nbatches_once = nbatches;
    for(STARSH_int batch_start = 0; batch_start < nbatches;
            batch_start += nbatches_once)
    {
        STARSH_int batch_end = batch_start + nbatches_once;
        if(batch_end > nbatches)
            batch_end = nbatches;
        for(bi = batch_start; bi < batch_end; ++bi)
        {
            //printf("RUNNING BATCH=%d\n", bi);
            STARSH_int this_batch_size = nblocks_far - bi*batch_size;
            if(this_batch_size > batch_size)
                this_batch_size = batch_size;
            // Run KBLAS_RSVD
            //*
            starpu_task_insert(&codelet_lowrank,
                    STARPU_VALUE, &this_batch_size, sizeof(this_batch_size),
                    STARPU_VALUE, &nb, sizeof(nb),
                    STARPU_VALUE, &maxrank, sizeof(maxrank),
                    STARPU_VALUE, &oversample, sizeof(oversample),
                    STARPU_VALUE, &tol, sizeof(tol),
                    STARPU_VALUE, &cuhandles, sizeof(cuhandles),
                    STARPU_VALUE, &khandles, sizeof(khandles),
                    STARPU_VALUE, &kstates, sizeof(kstates),
                    STARPU_VALUE, &wwork, sizeof(wwork),
                    STARPU_VALUE, &lwork, sizeof(lwork),
                    STARPU_VALUE, &wiwork, sizeof(wiwork),
                    STARPU_R, D_handle[bi],
                    STARPU_SCRATCH, Dcopy_handle[bi],
                    STARPU_W, U_handle[bi],
                    STARPU_W, V_handle[bi],
                    STARPU_W, rank_handle[bi],
                    0);
            starpu_data_unregister_submit(Dcopy_handle[bi]);
            starpu_task_insert(&codelet_get_data_back_to_cpu,
                    STARPU_R, U_handle[bi],
                    0);
            starpu_task_insert(&codelet_get_data_back_to_cpu,
                    STARPU_R, V_handle[bi],
                    0);
            starpu_task_insert(&codelet_get_data_back_to_cpu,
                    STARPU_R, rank_handle[bi],
                    0);
            starpu_data_unregister_submit(rank_handle[bi]);
            starpu_data_unregister_submit(D_handle[bi]);
            starpu_data_unregister_submit(U_handle[bi]);
            starpu_data_unregister_submit(V_handle[bi]);
        }
        //starpu_task_wait_for_all();
    }
    //time1 = omp_get_wtime();
    //printf("COMPRESS MATRIX IN: %f seconds\n", time1-time0);
    //time0 = time1;
    //printf("FINISH FIRST PASS AND UNREGISTER IN: %f seconds\n",
    //        omp_get_wtime()-time0);
    // Get number of false far-field blocks
    STARSH_int nblocks_false_far = 0;
    STARSH_int *false_far = NULL;
    for(bi = 0; bi < nblocks_far; bi++)
    {
        //printf("FAR_RANK[%zu]=%d\n", bi, far_rank[bi]);
        //far_rank[bi] = 0;
        if(far_rank[bi] == -1)
            nblocks_false_far++;
    }
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
        STARSH_MALLOC(near_D, new_nblocks_near);
        size_t size_D = new_nblocks_near * nb * nb;
        STARSH_MALLOC(alloc_D, size_D);
        nbatches = (new_nblocks_near-1)/batch_size + 1;
        starpu_data_handle_t D_handle[nbatches];
        starpu_data_handle_t index_handle[nbatches];
        int shape[] = {nb, nb};
        // For each near-field block compute its elements
        for(bi = 0; bi < new_nblocks_near; ++bi)
        {
            // Get indexes of corresponding block row and block column
            //STARSH_int i = block_near[2*bi];
            //STARSH_int j = block_near[2*bi+1];
            array_from_buffer(near_D+bi, 2, shape, 'd', 'F',
                    alloc_D + bi*nb*nb);
        }
        for(bi = 0; bi < nbatches; ++bi)
        {
            STARSH_int this_batch_size = new_nblocks_near - bi*batch_size;
            if(this_batch_size > batch_size)
                this_batch_size = batch_size;
            STARSH_int D_size = this_batch_size * nb * nb;
            double *D = alloc_D + bi*batch_size*nb*nb;
            starpu_vector_data_register(D_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(D), D_size, sizeof(*D));
            starpu_vector_data_register(index_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(block_near + 2*bi*batch_size),
                    2*this_batch_size, sizeof(*block_near));
        }
        for(bi = 0; bi < nbatches; ++bi)
        {
            STARSH_int this_batch_size = new_nblocks_near - bi*batch_size;
            if(this_batch_size > batch_size)
                this_batch_size = batch_size;
            // Generate matrix
            starpu_task_insert(&codelet_kernel,
                    STARPU_VALUE, &F, sizeof(F),
                    STARPU_VALUE, &this_batch_size, sizeof(this_batch_size),
                    STARPU_W, D_handle[bi],
                    STARPU_R, index_handle[bi],
                    0);
        }
        // Wait in this scope, because all handles are not visible outside
        starpu_task_wait_for_all();
        // Unregister data
        for(bi = 0; bi < nbatches; bi++)
        {
            starpu_data_unregister(D_handle[bi]);
            starpu_data_unregister(index_handle[bi]);
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
    starpu_execute_on_each_worker(deinit_starpu_kblas, args_gpu, STARPU_CUDA);
    starpu_execute_on_each_worker(deinit_starpu_cpu, args_cpu, STARPU_CPU);
    return starsh_blrm_new(matrix, F, far_rank, far_U, far_V, onfly, near_D,
            alloc_U, alloc_V, alloc_D, '1');
}

