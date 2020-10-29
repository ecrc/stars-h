/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/starpu/blrm/dmml.c
 *
 * @cond
 * This command in pair with endcond will prevent file from being documented.
 *
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "starsh.h"
#include "common.h"
#include "starsh-starpu.h"

// Compilation of this file should file because it is intended to fail
// This fail is commented due to bug

//int starsh_blrm__dmml_starpu(STARSH_blrm *matrix, int nrhs, double alpha,
//        double *A, int lda, double beta, double *B, int ldb)
//! Multiply blr-matrix by dense matrix.
/*! Performs `C=alpha*A*B+beta*C` with @ref STARSH_blrm `A` and dense matrices
 * `B` and `C`. All the integer types are int, since they are used in BLAS
 * calls.
 *
 * @param[in] matrix: Pointer to @ref STARSH_blrm object.
 * @param[in] nrhs: Number of right hand sides.
 * @param[in] alpha: Scalar mutliplier.
 * @param[in] A: Dense matrix, right havd side.
 * @param[in] lda: Leading dimension of `A`.
 * @param[in] beta: Scalar multiplier.
 * @param[in] B: Resulting dense matrix.
 * @param[in] ldb: Leading dimension of B.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup blrm
 * */
{
    STARSH_blrm *M = matrix;
    STARSH_blrf *F = M->format;
    STARSH_problem *P = F->problem;
    STARSH_kernel *kernel = P->kernel;
    STARSH_int nrows = P->shape[0];
    STARSH_int ncols = P->shape[P->ndim-1];
    // Shorcuts to information about clusters
    STARSH_cluster *R = F->row_cluster;
    STARSH_cluster *C = F->col_cluster;
    void *RD = R->data, *CD = C->data;
    // Number of far-field and near-field blocks
    STARSH_int nblocks_far = F->nblocks_far;
    STARSH_int nblocks_near = F->nblocks_near;
    STARSH_int bi;
    int T = CblasTrans, N = CblasNoTrans;
    double one = 1.0, zero = 0.0;
    char symm = F->symm;
    // Setting B = beta*B
    if(beta == 0.)
        for(size_t i = 0; i < nrhs; i++)
            for(size_t j = 0; j < nrows; j++)
                B[i*ldb+j] = 0.;
    else
        for(size_t i = 0; i < nrhs; i++)
            for(size_t j = 0; j < nrows; j++)
                B[i*ldb+j] *= beta;
    starpu_data_handle_t U_handle[nblocks_far], V_handle[nblocks_far];
    starpu_data_handle_t A_handle[nblocks_far], B_handle[nblocks_far];
    starpu_data_handle_t work_handle[nblocks_far];
    starpu_data_handle_t A2_handle[nblocks_far], B2_handle[nblocks_far];
    starpu_data_handle_t work2_handle[nblocks_far];
    struct starpu_codelet codelet =
    {
        .cpu_funcs = {starsh_dense_dgemm_starpu},
        .nbuffers = 3,
        .modes = {STARPU_R, STARPU_R, STARPU_RW}
    };
    struct starpu_codelet codelet_kernel =
    {
        .cpu_funcs = {starsh_dense_kernel_starpu},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W}
    };
    struct starpu_codelet fake_codelet =
    {
        .cpu_funcs = {starsh_dense_fake_init_starpu},
        .nbuffers = 1,
        .modes = {STARPU_W}
    };
    // Simple cycle over all far-field admissible blocks
    for(bi = 0; bi < nblocks_far; bi++)
    {
        // Get indexes of corresponding block row and block column
        STARSH_int i = F->block_far[2*bi];
        STARSH_int j = F->block_far[2*bi+1];
        // Get sizes and rank in int type due to BLAS calls
        int nrows = R->size[i];
        int ncols = C->size[j];
        int rank = M->far_rank[bi];
        // Get pointers to data buffers
        double *U = M->far_U[bi]->data, *V = M->far_V[bi]->data;
        // Register data
        starpu_vector_data_register(U_handle+bi, STARPU_MAIN_RAM,
                (uintptr_t)U, nrows*(size_t)rank, sizeof(*U));
        starpu_vector_data_register(V_handle+bi, STARPU_MAIN_RAM,
                (uintptr_t)V, ncols*(size_t)rank, sizeof(*V));
        starpu_vector_data_register(work_handle+bi, -1, 0, (size_t)nrhs*rank,
                sizeof(*U));
        starpu_vector_data_register(B_handle+bi, STARPU_MAIN_RAM,
                (uintptr_t)(B+R->start[i]),
                nrhs*(size_t)ldb-R->start[i], sizeof(*B));
        starpu_vector_data_register(A_handle+bi, STARPU_MAIN_RAM,
                (uintptr_t)(A+C->start[j]),
                nrhs*(size_t)lda-C->start[j], sizeof(*A));
        // fake task to make workhandle initialized
        starpu_task_insert(&fake_codelet, STARPU_W, work_handle[bi], 0);
        // Multiply low-rank matrix in U*V^T format by a dense matrix
        starpu_task_insert(&codelet, STARPU_VALUE, &T, sizeof(T),
                STARPU_VALUE, &N, sizeof(N),
                STARPU_VALUE, &rank, sizeof(rank),
                STARPU_VALUE, &nrhs, sizeof(nrhs),
                STARPU_VALUE, &ncols, sizeof(ncols),
                STARPU_VALUE, &one, sizeof(one),
                STARPU_R, V_handle[bi],
                STARPU_VALUE, &ncols, sizeof(ncols),
                STARPU_R, A_handle[bi],
                STARPU_VALUE, &lda, sizeof(lda),
                STARPU_VALUE, &zero, sizeof(zero),
                STARPU_RW, work_handle[bi],
                STARPU_VALUE, &rank, sizeof(rank),
                0);
        //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
        //        ncols, 1.0, V, ncols, A+C->start[j], lda, 0.0, D, rank);
        starpu_task_insert(&codelet, STARPU_VALUE, &N, sizeof(N),
                STARPU_VALUE, &N, sizeof(N),
                STARPU_VALUE, &nrows, sizeof(nrows),
                STARPU_VALUE, &nrhs, sizeof(nrhs),
                STARPU_VALUE, &rank, sizeof(rank),
                STARPU_VALUE, &alpha, sizeof(alpha),
                STARPU_R, U_handle[bi],
                STARPU_VALUE, &nrows, sizeof(nrows),
                STARPU_R, work_handle[bi],
                STARPU_VALUE, &rank, sizeof(rank),
                STARPU_VALUE, &one, sizeof(one),
                STARPU_RW, B_handle[bi],
                STARPU_VALUE, &ldb, sizeof(ldb),
                0);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, nrhs,
        //        rank, alpha, U, nrows, D, rank, 1.0, B+R->start[i], ldb);
        starpu_data_unregister_submit(work_handle[bi]);
        if(i != j && symm == 'S')
        {
            // Multiply low-rank matrix in V*U^T format by a dense matrix
            // U and V are simply swapped in case of symmetric block
            starpu_vector_data_register(B2_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(B+C->start[j]),
                    nrhs*(size_t)ldb-C->start[j], sizeof(*B));
            starpu_vector_data_register(A2_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(A+R->start[i]),
                    nrhs*(size_t)lda-R->start[i], sizeof(*A));
            starpu_vector_data_register(work2_handle+bi, -1, 0,
                    (size_t)nrhs*rank, sizeof(*U));
            // fake task to make workhandle initialized
            starpu_task_insert(&fake_codelet, STARPU_W, work2_handle[bi], 0);
            starpu_task_insert(&codelet, STARPU_VALUE, &T, sizeof(T),
                    STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &rank, sizeof(rank),
                    STARPU_VALUE, &nrhs, sizeof(nrhs),
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_R, U_handle[bi],
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_R, A2_handle[bi],
                    STARPU_VALUE, &lda, sizeof(lda),
                    STARPU_VALUE, &zero, sizeof(zero),
                    STARPU_RW, work2_handle[bi],
                    STARPU_VALUE, &rank, sizeof(rank),
                    0);
            //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
            //        nrows, 1.0, U, nrows, A+R->start[i], lda, 0.0, D, rank);
            starpu_task_insert(&codelet, STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &ncols, sizeof(ncols),
                    STARPU_VALUE, &nrhs, sizeof(nrhs),
                    STARPU_VALUE, &rank, sizeof(rank),
                    STARPU_VALUE, &alpha, sizeof(alpha),
                    STARPU_R, V_handle[bi],
                    STARPU_VALUE, &ncols, sizeof(ncols),
                    STARPU_R, work2_handle[bi],
                    STARPU_VALUE, &rank, sizeof(rank),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_RW, B2_handle[bi],
                    STARPU_VALUE, &ldb, sizeof(ldb),
                    0);
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ncols,
            //        nrhs, rank, alpha, V, ncols, D, rank, 1.0,
            //        B+C->start[j], ldb);
            starpu_data_unregister_submit(A2_handle[bi]);
            starpu_data_unregister_submit(B2_handle[bi]);
            starpu_data_unregister_submit(work2_handle[bi]);
        }
        starpu_data_unregister_submit(U_handle[bi]);
        starpu_data_unregister_submit(V_handle[bi]);
        starpu_data_unregister_submit(A_handle[bi]);
        starpu_data_unregister_submit(B_handle[bi]);
    }
    starpu_task_wait_for_all();
    starpu_data_handle_t A3_handle[nblocks_near], B3_handle[nblocks_near];
    starpu_data_handle_t work3_handle[nblocks_near];
    starpu_data_handle_t A4_handle[nblocks_near], B4_handle[nblocks_near];
    starpu_data_handle_t work4_handle[nblocks_near];
    starpu_data_handle_t bi_handle[nblocks_near];
    STARSH_int bi_value[nblocks_near];
    if(M->onfly == 1)
        // Simple cycle over all near-field blocks
        for(bi = 0; bi < nblocks_near; bi++)
        {
            // Get indexes and sizes of corresponding block row and column
            STARSH_int i = F->block_near[2*bi];
            STARSH_int j = F->block_near[2*bi+1];
            // Get sizes in int type due to BLAS calls
            int nrows = R->size[i];
            int ncols = C->size[j];
            // Register data
            bi_value[bi] = bi;
            starpu_variable_data_register(bi_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(bi_value+bi), sizeof(*bi_value));
            starpu_vector_data_register(work3_handle+bi, -1, 0,
                    (size_t)nrows*ncols, sizeof(*A));
            starpu_vector_data_register(B3_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(B+R->start[i]),
                    nrhs*(size_t)ldb-R->start[i], sizeof(*B));
            starpu_vector_data_register(A3_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(A+C->start[j]),
                    nrhs*(size_t)lda-C->start[j], sizeof(*A));
            // Fill temporary buffer with elements of corresponding block
            starpu_task_insert(&codelet_kernel, STARPU_VALUE, &F, sizeof(F),
                    STARPU_R, bi_handle[bi], STARPU_W, work3_handle[bi], 0);
            starpu_data_unregister_submit(bi_handle[bi]);
            //kernel(nrows, ncols, R->pivot+R->start[i], C->pivot+C->start[j],
            //        RD, CD, D, nrows);
            // Multiply 2 dense matrices
            starpu_task_insert(&codelet, STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_VALUE, &nrhs, sizeof(nrhs),
                    STARPU_VALUE, &ncols, sizeof(ncols),
                    STARPU_VALUE, &alpha, sizeof(alpha),
                    STARPU_R, work3_handle[bi],
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_R, A3_handle[bi],
                    STARPU_VALUE, &lda, sizeof(lda),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_RW, B3_handle[bi],
                    STARPU_VALUE, &ldb, sizeof(ldb),
                    0);
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
            //        nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
            //        B+R->start[i], ldb);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                starpu_vector_data_register(B4_handle+bi, STARPU_MAIN_RAM,
                        (uintptr_t)(B+C->start[j]),
                        nrhs*(size_t)ldb-C->start[j], sizeof(*B));
                starpu_vector_data_register(A4_handle+bi, STARPU_MAIN_RAM,
                        (uintptr_t)(A+R->start[i]),
                        nrhs*(size_t)lda-R->start[i], sizeof(*A));
                starpu_task_insert(&codelet, STARPU_VALUE, &T, sizeof(T),
                        STARPU_VALUE, &N, sizeof(N),
                        STARPU_VALUE, &ncols, sizeof(ncols),
                        STARPU_VALUE, &nrhs, sizeof(nrhs),
                        STARPU_VALUE, &nrows, sizeof(nrows),
                        STARPU_VALUE, &alpha, sizeof(alpha),
                        STARPU_R, work3_handle[bi],
                        STARPU_VALUE, &nrows, sizeof(nrows),
                        STARPU_R, A4_handle[bi],
                        STARPU_VALUE, &lda, sizeof(lda),
                        STARPU_VALUE, &one, sizeof(one),
                        STARPU_RW, B4_handle[bi],
                        STARPU_VALUE, &ldb, sizeof(ldb),
                        0);
                //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                //        ncols, nrhs, nrows, alpha, D, nrows, A+R->start[i],
                //        lda, 1.0, B+C->start[j], ldb);
                starpu_data_unregister_submit(A4_handle[bi]);
                starpu_data_unregister_submit(B4_handle[bi]);
            }
            starpu_data_unregister_submit(A3_handle[bi]);
            starpu_data_unregister_submit(B3_handle[bi]);
            starpu_data_unregister_submit(work3_handle[bi]);
        }
    if(M->onfly == 0)
        // Simple cycle over all near-field blocks
        for(bi = 0; bi < nblocks_near; bi++)
        {
            // Get indexes and sizes of corresponding block row and column
            STARSH_int i = F->block_near[2*bi];
            STARSH_int j = F->block_near[2*bi+1];
            // Get sizes in int type due to BLAS calls
            int nrows = R->size[i];
            int ncols = C->size[j];
            // Get pointers to data buffers
            double *D = M->near_D[bi]->data;
            // Register data
            starpu_vector_data_register(work3_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)D, (size_t)nrows*ncols, sizeof(*D));
            starpu_vector_data_register(B3_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(B+R->start[i]),
                    nrhs*(size_t)ldb-R->start[i], sizeof(*B));
            starpu_vector_data_register(A3_handle+bi, STARPU_MAIN_RAM,
                    (uintptr_t)(A+C->start[j]),
                    nrhs*(size_t)lda-C->start[j], sizeof(*A));
            // Multiply 2 dense matrices
            starpu_task_insert(&codelet, STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_VALUE, &nrhs, sizeof(nrhs),
                    STARPU_VALUE, &ncols, sizeof(ncols),
                    STARPU_VALUE, &alpha, sizeof(alpha),
                    STARPU_R, work3_handle[bi],
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_R, A3_handle[bi],
                    STARPU_VALUE, &lda, sizeof(lda),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_RW, B3_handle[bi],
                    STARPU_VALUE, &ldb, sizeof(ldb),
                    0);
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
            //        nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
            //        B+R->start[i], ldb);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                starpu_vector_data_register(B4_handle+bi, STARPU_MAIN_RAM,
                        (uintptr_t)(B+C->start[j]),
                        nrhs*(size_t)ldb-C->start[j], sizeof(*B));
                starpu_vector_data_register(A4_handle+bi, STARPU_MAIN_RAM,
                        (uintptr_t)(A+R->start[i]),
                        nrhs*(size_t)lda-R->start[i], sizeof(*A));
                starpu_task_insert(&codelet, STARPU_VALUE, &T, sizeof(T),
                        STARPU_VALUE, &N, sizeof(N),
                        STARPU_VALUE, &ncols, sizeof(ncols),
                        STARPU_VALUE, &nrhs, sizeof(nrhs),
                        STARPU_VALUE, &nrows, sizeof(nrows),
                        STARPU_VALUE, &alpha, sizeof(alpha),
                        STARPU_R, work3_handle[bi],
                        STARPU_VALUE, &nrows, sizeof(nrows),
                        STARPU_R, A4_handle[bi],
                        STARPU_VALUE, &lda, sizeof(lda),
                        STARPU_VALUE, &one, sizeof(one),
                        STARPU_RW, B4_handle[bi],
                        STARPU_VALUE, &ldb, sizeof(ldb),
                        0);
                //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                //        ncols, nrhs, nrows, alpha, D, nrows, A+R->start[i],
                //        lda, 1.0, B+C->start[j], ldb);
                starpu_data_unregister_submit(A4_handle[bi]);
                starpu_data_unregister_submit(B4_handle[bi]);
            }
            starpu_data_unregister_submit(A3_handle[bi]);
            starpu_data_unregister_submit(B3_handle[bi]);
            starpu_data_unregister_submit(work3_handle[bi]);
        }
    starpu_task_wait_for_all();
    return 0;
}

//! @endcond
