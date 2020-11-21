/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/mpi_starpu_kblas/blrm/drsdd.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-starpu-kblas.h"
#include "starsh-mpi-starpu-kblas.h"
#include "starsh-spatial.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <kblas.h>
#include "batch_rand.h"
#include <starpu.h>
#include <mpi.h>

static void init_starpu_kblas(void *args)
{
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    kblasRandState_t *kblas_states;
    STARSH_ssdata **data_gpu;
    STARSH_ssdata *data_cpu;
    //double time0 = MPI_Wtime();
    cudaStream_t stream = starpu_cuda_get_local_stream();
    int nb, nsamples, maxbatch;
    starpu_codelet_unpack_args(args, &cublas_handles, &kblas_handles,
            &kblas_states, &data_gpu, &data_cpu, &nb, &nsamples, &maxbatch);
    int id = starpu_worker_get_id();
    cublasStatus_t status;
    //printf("unpack_args: %f seconds\n", MPI_Wtime()-time0);
    //time0 = MPI_Wtime();
    kblasCreate(&kblas_handles[id]);
    //printf("kblasCreate: %f seconds\n", MPI_Wtime()-time0);
    //time0 = MPI_Wtime();
    kblasSetStream(kblas_handles[id], stream);
    kblasDrsvd_batch_wsquery(kblas_handles[id], nb, nb, nsamples, maxbatch);
    kblasAllocateWorkspace(kblas_handles[id]);
    //printf("kblasAllocateWorkspace: %f seconds\n", MPI_Wtime()-time0);
    //time0 = MPI_Wtime();
    cublas_handles[id] = kblasGetCublasHandle(kblas_handles[id]);
    kblasInitRandState(kblas_handles[id], &kblas_states[id], 16384*2, 0);
    starsh_ssdata_togpu(&data_gpu[id], data_cpu);
    cudaStreamSynchronize(stream);
    //printf("starsh_ssdata_togpu: %f seconds\n", MPI_Wtime()-time0);
    //time0 = MPI_Wtime();
}

static void deinit_starpu_kblas(void *args)
{
    int nb, nsamples, maxbatch;
    cublasHandle_t *cublas_handles;
    kblasHandle_t *kblas_handles;
    kblasRandState_t *kblas_states;
    STARSH_ssdata **data_gpu;
    STARSH_ssdata *data_cpu;
    starpu_codelet_unpack_args(args, &cublas_handles, &kblas_handles,
            &kblas_states, &data_gpu, &data_cpu, &nb, &nsamples, &maxbatch);
    int id = starpu_worker_get_id();
    kblasDestroyRandState(kblas_states[id]);
    kblasDestroy(&kblas_handles[id]);
    starsh_ssdata_free_gpu(data_gpu[id]);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

static void starsh_dense_dlrrsdd_starpu_kblas3_copy(void *buffers[], void *cl_arg)
{
    int N, batch_size;
    starpu_codelet_unpack_args(cl_arg, &N, &batch_size);
    double *Dcopy = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *D = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    cblas_dcopy(N*N*batch_size, Dcopy, 1, D, 1);
}

int starsh_blrm__drsdd_mpi_starpu_kblas3_spatial(STARSH_blrm **matrix,
        STARSH_blrf *format, int maxrank, double tol, int onfly)
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
    double time_start = MPI_Wtime();
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
    // Temporary holder for indexes of tiles
    STARSH_int *tile_index = NULL;
    // Places to store low-rank factors, dense blocks and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank = NULL;
    double *alloc_U = NULL, *alloc_V = NULL, *alloc_D = NULL, *alloc_S = NULL;
    size_t offset_U = 0, offset_V = 0, offset_D = 0;
    STARSH_int lbi, lbj, bi, bj = 0;
    const int oversample = starsh_params.oversample;
    // MPI
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    //if(mpi_rank == 0)
    //    printf("MPIKBLAS3\n");
    // Init CuBLAS and KBLAS handles and temp buffers for all workers (but they
    // are used only in GPU codelets)
    int workers = starpu_worker_get_count();
    cublasHandle_t cublas_handles[workers];
    kblasHandle_t kblas_handles[workers];
    kblasRandState_t kblas_states[workers];
    STARSH_ssdata *data_gpu_array[workers];
    cublasHandle_t *cuhandles = cublas_handles;
    kblasHandle_t *khandles = kblas_handles;
    kblasRandState_t *kstates = kblas_states;
    STARSH_ssdata **data_gpu = data_gpu_array;
    //printf("MAIN: %p, %p, %p\n", cuhandles, khandles, svhandles);
    void *args_gpu;
    size_t args_gpu_size = 0;
    // This works only for TLR with equal tiles
    int nb = RC->size[0];
    int nsamples = maxrank+oversample;
    // Set size of batch
    char *env_var = getenv("STARSH_KBLAS_BATCH");
    int batch_size = 300;
    if(env_var)
        batch_size = atoi(env_var);
    //if(mpi_rank == 0)
    //    printf("MPIKBLAS3: batch_size=%d\n", batch_size);
    // Ceil number of batches
    int nbatches_local = (nblocks_far_local-1)/batch_size + 1;
    // Get number of temporary buffers for CPU-GPU transfers
    int nworkers_gpu = 3 * starpu_cuda_worker_get_count();
    // Get corresponding sizes and minimum of them
    int mn = maxrank+oversample;
    if(mn > nb)
        mn = nb;
    starpu_codelet_pack_args(&args_gpu, &args_gpu_size,
            STARPU_VALUE, &cuhandles, sizeof(cuhandles),
            STARPU_VALUE, &khandles, sizeof(khandles),
            STARPU_VALUE, &kstates, sizeof(kstates),
            STARPU_VALUE, &data_gpu, sizeof(data_gpu),
            STARPU_VALUE, &RD, sizeof(RD),
            STARPU_VALUE, &nb, sizeof(nb),
            STARPU_VALUE, &nsamples, sizeof(nsamples),
            STARPU_VALUE, &batch_size, sizeof(batch_size),
            0);
    starpu_execute_on_each_worker(init_starpu_kblas, args_gpu, STARPU_CUDA);
    //MPI_Barrier(MPI_COMM_WORLD);
    //double time0 = MPI_Wtime();
    //if(mpi_rank == 0)
    //    printf("CUBLAS + WORKSPACE ALLOCATION: %f seconds\n", time0-time_start);
    // Init codelet structs and handles
    struct starpu_codelet codelet_kernel =
    {
        .cuda_funcs = {starsh_dense_kernel_starpu_kblas3_gpu},
        .cuda_flags = {STARPU_CUDA_ASYNC},
        .nbuffers = 2,
        .modes = {STARPU_W, STARPU_R},
        //.type = STARPU_SPMD,
        //.max_parallelism = INT_MAX,
    };
    struct starpu_codelet codelet_lowrank =
    {
        .cuda_funcs = {starsh_dense_dlrrsdd_starpu_kblas2_gpu},
        .cuda_flags = {STARPU_CUDA_ASYNC},
        .nbuffers = 5,
        .modes = {STARPU_R, STARPU_SCRATCH, STARPU_W, STARPU_W, STARPU_W},
    };
    struct starpu_codelet codelet_getrank =
    {
        .cpu_funcs = {starsh_dense_dlrrsdd_starpu_kblas2_getrank},
        .nbuffers = 6,
        .modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_W, STARPU_W, STARPU_W},
        //.type = STARPU_SPMD,
        //.max_parallelism = INT_MAX,
    };
    struct starpu_codelet codelet_copy =
    {
        .cpu_funcs = {starsh_dense_dlrrsdd_starpu_kblas3_copy},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W},
    };
    //starpu_data_handle_t D_handle[nbatches_local];
    //starpu_data_handle_t Dcopy_handle[nbatches_local];
    starpu_data_handle_t index_handle[nbatches_local];
    starpu_data_handle_t U_handle[nbatches_local];
    starpu_data_handle_t V_handle[nbatches_local];
    //starpu_data_handle_t S_handle[nbatches_local];
    starpu_data_handle_t rank_handle[nbatches_local];
    starpu_data_handle_t D_handle[nworkers_gpu];
    starpu_data_handle_t Dcopy_handle[nworkers_gpu];
    starpu_data_handle_t tmp_U_handle[nworkers_gpu];
    starpu_data_handle_t tmp_V_handle[nworkers_gpu];
    starpu_data_handle_t tmp_S_handle[nworkers_gpu];
    // Init buffers to store low-rank factors of far-field blocks if needed
    MPI_Barrier(MPI_COMM_WORLD);
    //double time0 = MPI_Wtime();
    //if(mpi_rank == 0)
    //    printf("MPIKBLAS3: init in %f seconds\n", time0-time_start);
    if(nbatches_local > 0)
    {
        STARSH_MALLOC(far_U, nblocks_far_local);
        STARSH_MALLOC(far_V, nblocks_far_local);
        STARSH_MALLOC(far_rank, nblocks_far_local);
        size_t size_U = nblocks_far_local * nb * maxrank;
        size_t size_V = size_U;
        //size_t size_D = nblocks_far_local * nb * nb;
        //size_t size_S = nblocks_far_local * mn;
        STARSH_MALLOC(alloc_U, size_U);
        STARSH_MALLOC(alloc_V, size_V);
        //starpu_memory_pin(alloc_U, size_U*sizeof(double));
        //starpu_memory_pin(alloc_V, size_V*sizeof(double));
        //starpu_malloc(&alloc_S, size_S*sizeof(double));
        //starpu_malloc(&alloc_D, size_D*sizeof(double));
        int shape[] = {nb, maxrank};
        for(lbi = 0; lbi < nblocks_far_local; ++lbi)
        {
            STARSH_int offset = lbi * nb * maxrank;
            array_from_buffer(far_U+lbi, 2, shape, 'd', 'F', alloc_U+offset);
            array_from_buffer(far_V+lbi, 2, shape, 'd', 'F', alloc_V+offset);
        }
        starpu_malloc(&tile_index, 2*nblocks_far_local*sizeof(*tile_index));
        for(bi = 0; bi < nblocks_far_local; ++bi)
        {
            STARSH_int ind = block_far_local[bi];
            tile_index[2*bi] = block_far[2*ind];
            tile_index[2*bi+1] = block_far[2*ind+1];
        }
        for(lbi = 0; lbi < nbatches_local; ++lbi)
        {
            STARSH_int offset = lbi * batch_size * nb * maxrank;
            //STARSH_int offset_S = lbi * batch_size * mn;
            double *U = alloc_U + offset;
            double *V = alloc_V + offset;
            //double *S = alloc_S + offset_S;
            //STARSH_int offset_D = lbi * batch_size * nb * nb;
            //double *D = alloc_D + offset_D;
            int this_batch_size = nblocks_far_local - lbi*batch_size;
            if(this_batch_size > batch_size)
                this_batch_size = batch_size;
            //STARSH_int D_size = this_batch_size * nb * nb;
            STARSH_int U_size = this_batch_size * nb * maxrank;
            STARSH_int V_size = U_size;
            //STARSH_int S_size = this_batch_size * mn;
            //printf("THIS BATCH SIZE=%d\n", this_batch_size);
            starpu_vector_data_register(rank_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(far_rank + lbi*batch_size), this_batch_size,
                    sizeof(*far_rank));
            //starpu_vector_data_register(D_handle+lbi, STARPU_MAIN_RAM,
            //        (uintptr_t)(D), D_size, sizeof(double));
            //starpu_vector_data_register(Dcopy_handle+lbi, -1, 0, D_size,
            //        sizeof(double));
            starpu_vector_data_register(index_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(tile_index + 2*lbi*batch_size),
                    2*this_batch_size, sizeof(*tile_index));
            starpu_vector_data_register(U_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(U), U_size, sizeof(*U));
            starpu_vector_data_register(V_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(V), V_size, sizeof(*V));
            //starpu_vector_data_register(S_handle+lbi, STARPU_MAIN_RAM,
            //        (uintptr_t)(S), S_size, sizeof(double));
        }
        STARSH_int D_size = batch_size * nb * nb;
        STARSH_int tmp_U_size = batch_size * nb * maxrank;
        STARSH_int tmp_S_size = batch_size * mn;
        for(bi = 0; bi < nworkers_gpu; ++bi)
        {
            starpu_vector_data_register(D_handle+bi, -1, 0, D_size,
                    sizeof(double));
            starpu_vector_data_register(Dcopy_handle+bi, -1, 0, D_size,
                    sizeof(double));
            starpu_vector_data_register(tmp_U_handle+bi, -1, 0, tmp_U_size,
                    sizeof(double));
            starpu_vector_data_register(tmp_V_handle+bi, -1, 0, tmp_U_size,
                    sizeof(double));
            starpu_vector_data_register(tmp_S_handle+bi, -1, 0, tmp_S_size,
                    sizeof(double));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //double time1 = MPI_Wtime();
    //if(mpi_rank == 0)
    //    printf("MPIKBLAS3: Register data in %f seconds\n", time1-time0);
    //time0 = time1;
    // Work variables
    int info;
    // START MEASURING TIME
    for(lbi = 0; lbi < nbatches_local; ++lbi)
    {
        //printf("RUNNING BATCH=%d\n", bi);
        int this_batch_size = nblocks_far_local - lbi*batch_size;
        if(this_batch_size > batch_size)
            this_batch_size = batch_size;
        // Generate matrix
        starpu_task_insert(&codelet_kernel,
                STARPU_VALUE, &data_gpu, sizeof(data_gpu),
                STARPU_VALUE, &nb, sizeof(nb),
                STARPU_VALUE, &this_batch_size, sizeof(this_batch_size),
                STARPU_W, D_handle[lbi % nworkers_gpu],
                STARPU_R, index_handle[lbi],
                STARPU_PRIORITY, -2,
                0);
        starpu_data_unregister_submit(index_handle[lbi]);
        // Run KBLAS_RSVD
        starpu_task_insert(&codelet_lowrank,
                STARPU_VALUE, &this_batch_size, sizeof(this_batch_size),
                STARPU_VALUE, &nb, sizeof(nb),
                STARPU_VALUE, &maxrank, sizeof(maxrank),
                STARPU_VALUE, &oversample, sizeof(oversample),
                STARPU_VALUE, &tol, sizeof(tol),
                STARPU_VALUE, &cuhandles, sizeof(cuhandles),
                STARPU_VALUE, &khandles, sizeof(khandles),
                STARPU_VALUE, &kstates, sizeof(kstates),
                STARPU_R, D_handle[lbi % nworkers_gpu],
                STARPU_SCRATCH, Dcopy_handle[lbi % nworkers_gpu],
                STARPU_W, tmp_U_handle[lbi % nworkers_gpu],
                STARPU_W, tmp_V_handle[lbi % nworkers_gpu],
                STARPU_W, tmp_S_handle[lbi % nworkers_gpu],
                STARPU_PRIORITY, 0,
                0);
        //starpu_data_unregister_submit(D_handle[lbi]);
        //starpu_data_unregister_submit(Dcopy_handle[lbi]);
        starpu_task_insert(&codelet_getrank,
                STARPU_VALUE, &this_batch_size, sizeof(this_batch_size),
                STARPU_VALUE, &nb, sizeof(nb),
                STARPU_VALUE, &maxrank, sizeof(maxrank),
                STARPU_VALUE, &oversample, sizeof(oversample),
                STARPU_VALUE, &tol, sizeof(tol),
                STARPU_R, tmp_U_handle[lbi % nworkers_gpu],
                STARPU_R, tmp_V_handle[lbi % nworkers_gpu],
                STARPU_R, tmp_S_handle[lbi % nworkers_gpu],
                STARPU_W, rank_handle[lbi],
                STARPU_W, U_handle[lbi],
                STARPU_W, V_handle[lbi],
                STARPU_PRIORITY, -1,
                0);
        starpu_data_unregister_submit(rank_handle[lbi]);
        starpu_data_unregister_submit(U_handle[lbi]);
        starpu_data_unregister_submit(V_handle[lbi]);
        //starpu_data_unregister_submit(S_handle[lbi]);
    }
    starpu_task_wait_for_all();
    MPI_Barrier(MPI_COMM_WORLD);
    //time1 = MPI_Wtime();
    //if(mpi_rank == 0)
    //    printf("COMPUTE+COMPRESS MATRIX IN: %f seconds\n", time1-time0);
    //time0 = time1;
    if(nbatches_local > 0)
    {
        //size_t size_U = nblocks_far_local * nb * maxrank;
        //size_t size_V = size_U;
        //starpu_free(alloc_D);
        //starpu_memory_unpin(alloc_U, size_U*sizeof(double));
        //starpu_memory_unpin(alloc_V, size_V*sizeof(double));
        //starpu_free(alloc_S);
        starpu_free(tile_index);
        for(bi = 0; bi < nworkers_gpu; ++bi)
        {
            starpu_data_unregister(D_handle[bi]);
            starpu_data_unregister(Dcopy_handle[bi]);
            starpu_data_unregister(tmp_U_handle[bi]);
            starpu_data_unregister(tmp_V_handle[bi]);
            starpu_data_unregister(tmp_S_handle[bi]);
        }
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(mpi_rank == 0)
    //    printf("FINISH FIRST PASS AND UNREGISTER IN: %f seconds\n",
    //            MPI_Wtime()-time0);
    // Get number of false far-field blocks
    STARSH_int nblocks_false_far_local = 0;
    STARSH_int *false_far_local = NULL;
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
    {
        //far_rank[lbi] = -1;
        if(far_rank[lbi] == -1)
            nblocks_false_far_local++;
    }
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
        STARSH_MALLOC(near_D, new_nblocks_near_local);
        size_t size_D = new_nblocks_near_local * nb * nb;
        STARSH_MALLOC(alloc_D, size_D);
        nbatches_local = (new_nblocks_near_local-1)/batch_size + 1;
        starpu_data_handle_t D_handle[nbatches_local];
        starpu_data_handle_t index_handle[nbatches_local];
        starpu_malloc(&tile_index, 2*new_nblocks_near_local*sizeof(*tile_index));
        int shape[] = {nb, nb};
        // For each local near-field block compute its elements
        for(lbi = 0; lbi < new_nblocks_near_local; ++lbi)
        {
            // Get indexes of corresponding block row and block column
            array_from_buffer(near_D+lbi, 2, shape, 'd', 'F',
                    alloc_D + lbi*nb*nb);
            STARSH_int ind = block_near_local[lbi];
            tile_index[lbi*2] = block_near[2*ind];
            tile_index[lbi*2+1] = block_near[2*ind+1];
        }
        for(lbi = 0; lbi < nbatches_local; ++lbi)
        {
            int this_batch_size = new_nblocks_near_local
                - lbi*batch_size;
            if(this_batch_size > batch_size)
                this_batch_size = batch_size;
            STARSH_int D_size = this_batch_size * nb * nb;
            double *D = alloc_D + lbi*batch_size*nb*nb;
            starpu_vector_data_register(D_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(D), D_size, sizeof(*D));
            starpu_vector_data_register(index_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(tile_index + 2*lbi*batch_size),
                    2*this_batch_size, sizeof(*tile_index));
        }
        for(lbi = 0; lbi < nbatches_local; ++lbi)
        {
            int this_batch_size = new_nblocks_near_local
                - lbi*batch_size;
            if(this_batch_size > batch_size)
                this_batch_size = batch_size;
            // Generate matrix
            starpu_task_insert(&codelet_kernel,
                    STARPU_VALUE, &data_gpu, sizeof(data_gpu),
                    STARPU_VALUE, &nb, sizeof(nb),
                    STARPU_VALUE, &this_batch_size, sizeof(this_batch_size),
                    STARPU_W, D_handle[lbi],
                    STARPU_R, index_handle[lbi],
                    0);
            starpu_data_unregister_submit(D_handle[lbi]);
            starpu_data_unregister_submit(index_handle[lbi]);
        }
        // Wait in this scope, because all handles are not visible outside
        starpu_task_wait_for_all();
        starpu_free(tile_index);
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
    //if(mpi_rank == 0)
    //    printf("FINISH NEAR-FIELD TILES: %f seconds\n", MPI_Wtime()-time0);
    //time0 = MPI_Wtime();
    starpu_execute_on_each_worker(deinit_starpu_kblas, args_gpu, STARPU_CUDA);
    //if(mpi_rank == 0)
    //    printf("MPIKBLAS3: finalize in %f seconds\n", MPI_Wtime()-time0);
    return starsh_blrm_new_mpi(matrix, F, far_rank, far_U, far_V, onfly,
            near_D, alloc_U, alloc_V, alloc_D, '1');
}

