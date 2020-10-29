/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/mpi_starpu/blrm/dmml.c
 *
 * @cond
 * This command in pair with endcond will prevent file from being documented.
 *
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"
#include "starsh-mpi-starpu.h"

// Compilation of this file should file because it is intended to fail
// This fail is commented due to bug

//int starsh_blrm__dmml_mpi_starpu(STARSH_blrm *matrix, int nrhs, double alpha,
//        double *A, int lda, double beta, double *B, int ldb)
//! Multiply blr-matrix by dense matrix on MPI nodes.
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
    STARSH_int nblocks_far_local = F->nblocks_far_local;
    STARSH_int nblocks_near_local = F->nblocks_near_local;
    STARSH_int lbi;
    int T = CblasTrans, N = CblasNoTrans;
    double one = 1.0, zero = 0.0;
    char symm = F->symm;
    int maxrank = 0;
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
        if(maxrank < M->far_rank[lbi])
            maxrank = M->far_rank[lbi];
    STARSH_int maxnb = nrows/F->nbrows;
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    for(int i = 0; i < nrhs; i++)
        MPI_Bcast(A+i*lda, ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double *temp_B;
    STARSH_MALLOC(temp_B, nrhs*nrows);
    for(size_t j = 0; j < nrhs*(size_t)nrows; j++)
        temp_B[j] = 0.;
    if(beta != 0. && mpi_rank == 0)
        for(STARSH_int i = 0; i < nrows; i++)
            for(STARSH_int j = 0; j < nrhs; j++)
                temp_B[j*nrows+i] = beta*B[j*ldb+i];
    int ldout = nrows;
    starpu_data_handle_t U_handle[nblocks_far_local];
    starpu_data_handle_t V_handle[nblocks_far_local];
    starpu_data_handle_t A_handle[nblocks_far_local];
    starpu_data_handle_t B_handle[nblocks_far_local];
    starpu_data_handle_t work_handle[nblocks_far_local];
    starpu_data_handle_t A2_handle[nblocks_far_local];
    starpu_data_handle_t B2_handle[nblocks_far_local];
    starpu_data_handle_t work2_handle[nblocks_far_local];
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
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
    {
        STARSH_int bi = F->block_far_local[lbi];
        // Get indexes of corresponding block row and block column
        STARSH_int i = F->block_far[2*bi];
        STARSH_int j = F->block_far[2*bi+1];
        // Get sizes and rank
        int nrows = R->size[i];
        int ncols = C->size[j];
        int rank = M->far_rank[lbi];
        if(rank == 0)
            continue;
        // Get pointers to data buffers
        double *U = M->far_U[lbi]->data, *V = M->far_V[lbi]->data;
        int info = 0;
        // Register data
        starpu_vector_data_register(U_handle+lbi, STARPU_MAIN_RAM,
                (uintptr_t)U, nrows*(size_t)rank, sizeof(*U));
        starpu_vector_data_register(V_handle+lbi, STARPU_MAIN_RAM,
                (uintptr_t)V, ncols*(size_t)rank, sizeof(*V));
        starpu_vector_data_register(work_handle+lbi, -1, 0, (size_t)nrhs*rank,
                sizeof(*U));
        starpu_vector_data_register(B_handle+lbi, STARPU_MAIN_RAM,
                (uintptr_t)(temp_B+R->start[i]),
                nrhs*(size_t)ldout-R->start[i], sizeof(*temp_B));
        starpu_vector_data_register(A_handle+lbi, STARPU_MAIN_RAM,
                (uintptr_t)(A+C->start[j]),
                nrhs*(size_t)lda-C->start[j], sizeof(*A));
        // fake task to make workhandle initialized
        starpu_task_insert(&fake_codelet, STARPU_W, work_handle[lbi], 0);
        // Multiply low-rank matrix in U*V^T format by a dense matrix
        starpu_task_insert(&codelet, STARPU_VALUE, &T, sizeof(T),
                STARPU_VALUE, &N, sizeof(N),
                STARPU_VALUE, &rank, sizeof(rank),
                STARPU_VALUE, &nrhs, sizeof(nrhs),
                STARPU_VALUE, &ncols, sizeof(ncols),
                STARPU_VALUE, &one, sizeof(one),
                STARPU_R, V_handle[lbi],
                STARPU_VALUE, &ncols, sizeof(ncols),
                STARPU_R, A_handle[lbi],
                STARPU_VALUE, &lda, sizeof(lda),
                STARPU_VALUE, &zero, sizeof(zero),
                STARPU_RW, work_handle[lbi],
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
                STARPU_R, U_handle[lbi],
                STARPU_VALUE, &nrows, sizeof(nrows),
                STARPU_R, work_handle[lbi],
                STARPU_VALUE, &rank, sizeof(rank),
                STARPU_VALUE, &one, sizeof(one),
                STARPU_RW, B_handle[lbi],
                STARPU_VALUE, &ldout, sizeof(ldout),
                0);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, nrhs,
        //        rank, alpha, U, nrows, D, rank, 1.0, out+R->start[i], ldout);
        starpu_data_unregister_submit(work_handle[lbi]);
        if(i != j && symm == 'S')
        {
            // Multiply low-rank matrix in V*U^T format by a dense matrix
            // U and V are simply swapped in case of symmetric block
            starpu_vector_data_register(B2_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(temp_B+C->start[j]),
                    nrhs*(size_t)ldout-C->start[j], sizeof(*temp_B));
            starpu_vector_data_register(A2_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(A+R->start[i]),
                    nrhs*(size_t)lda-R->start[i], sizeof(*A));
            starpu_vector_data_register(work2_handle+lbi, -1, 0,
                    (size_t)nrhs*rank, sizeof(*U));
            // fake task to make workhandle initialized
            starpu_task_insert(&fake_codelet, STARPU_W, work2_handle[lbi], 0);
            starpu_task_insert(&codelet, STARPU_VALUE, &T, sizeof(T),
                    STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &rank, sizeof(rank),
                    STARPU_VALUE, &nrhs, sizeof(nrhs),
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_R, U_handle[lbi],
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_R, A2_handle[lbi],
                    STARPU_VALUE, &lda, sizeof(lda),
                    STARPU_VALUE, &zero, sizeof(zero),
                    STARPU_RW, work2_handle[lbi],
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
                    STARPU_R, V_handle[lbi],
                    STARPU_VALUE, &ncols, sizeof(ncols),
                    STARPU_R, work2_handle[lbi],
                    STARPU_VALUE, &rank, sizeof(rank),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_RW, B2_handle[lbi],
                    STARPU_VALUE, &ldout, sizeof(ldout),
                    0);
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ncols,
            //        nrhs, rank, alpha, V, ncols, D, rank, 1.0,
            //        out+C->start[j], ldout);
            starpu_data_unregister_submit(A2_handle[lbi]);
            starpu_data_unregister_submit(B2_handle[lbi]);
            starpu_data_unregister_submit(work2_handle[lbi]);
        }
        starpu_data_unregister_submit(U_handle[lbi]);
        starpu_data_unregister_submit(V_handle[lbi]);
        starpu_data_unregister_submit(A_handle[lbi]);
        starpu_data_unregister_submit(B_handle[lbi]);
    }
    starpu_task_wait_for_all();
    starpu_data_handle_t A3_handle[nblocks_near_local];
    starpu_data_handle_t B3_handle[nblocks_near_local];
    starpu_data_handle_t work3_handle[nblocks_near_local];
    starpu_data_handle_t A4_handle[nblocks_near_local];
    starpu_data_handle_t B4_handle[nblocks_near_local];
    starpu_data_handle_t work4_handle[nblocks_near_local];
    starpu_data_handle_t bi_handle[nblocks_near_local];
    STARSH_int bi_value[nblocks_near_local];
    if(M->onfly == 1)
        // Simple cycle over all near-field blocks
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
        {
            STARSH_int bi = F->block_near_local[lbi];
            // Get indexes and sizes of corresponding block row and column
            STARSH_int i = F->block_near[2*bi];
            STARSH_int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            int info = 0;
            // Register data
            bi_value[lbi] = bi;
            starpu_variable_data_register(bi_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(bi_value+lbi), sizeof(*bi_value));
            starpu_vector_data_register(work3_handle+lbi, -1, 0,
                    (size_t)nrows*ncols, sizeof(*A));
            starpu_vector_data_register(B3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(temp_B+R->start[i]),
                    nrhs*(size_t)ldout-R->start[i], sizeof(*temp_B));
            starpu_vector_data_register(A3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(A+C->start[j]),
                    nrhs*(size_t)lda-C->start[j], sizeof(*A));
            // Fill temporary buffer with elements of corresponding block
            starpu_task_insert(&codelet_kernel, STARPU_VALUE, &F, sizeof(F),
                    STARPU_R, bi_handle[lbi], STARPU_W, work3_handle[lbi], 0);
            starpu_data_unregister_submit(bi_handle[lbi]);
            //kernel(nrows, ncols, R->pivot+R->start[i],
            //        C->pivot+C->start[j], RD, CD, D, nrows);
            // Multiply 2 dense matrices
            starpu_task_insert(&codelet, STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_VALUE, &nrhs, sizeof(nrhs),
                    STARPU_VALUE, &ncols, sizeof(ncols),
                    STARPU_VALUE, &alpha, sizeof(alpha),
                    STARPU_R, work3_handle[lbi],
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_R, A3_handle[lbi],
                    STARPU_VALUE, &lda, sizeof(lda),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_RW, B3_handle[lbi],
                    STARPU_VALUE, &ldout, sizeof(ldout),
                    0);
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
            //        nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
            //        out+R->start[i], ldout);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                starpu_vector_data_register(B4_handle+lbi, STARPU_MAIN_RAM,
                        (uintptr_t)(temp_B+C->start[j]),
                        nrhs*(size_t)ldout-C->start[j], sizeof(*temp_B));
                starpu_vector_data_register(A4_handle+lbi, STARPU_MAIN_RAM,
                        (uintptr_t)(A+R->start[i]),
                        nrhs*(size_t)lda-R->start[i], sizeof(*A));
                starpu_task_insert(&codelet, STARPU_VALUE, &T, sizeof(T),
                        STARPU_VALUE, &N, sizeof(N),
                        STARPU_VALUE, &ncols, sizeof(ncols),
                        STARPU_VALUE, &nrhs, sizeof(nrhs),
                        STARPU_VALUE, &nrows, sizeof(nrows),
                        STARPU_VALUE, &alpha, sizeof(alpha),
                        STARPU_R, work3_handle[lbi],
                        STARPU_VALUE, &nrows, sizeof(nrows),
                        STARPU_R, A4_handle[lbi],
                        STARPU_VALUE, &lda, sizeof(lda),
                        STARPU_VALUE, &one, sizeof(one),
                        STARPU_RW, B4_handle[lbi],
                        STARPU_VALUE, &ldout, sizeof(ldout),
                        0);
                //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncols,
                //        nrhs, nrows, alpha, D, nrows, A+R->start[i], lda,
                //        1.0, out+C->start[j], ldout);
                starpu_data_unregister_submit(A4_handle[lbi]);
                starpu_data_unregister_submit(B4_handle[lbi]);
            }
            starpu_data_unregister_submit(A3_handle[lbi]);
            starpu_data_unregister_submit(B3_handle[lbi]);
            starpu_data_unregister_submit(work3_handle[lbi]);
        }
    else
        // Simple cycle over all near-field blocks
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
        {
            STARSH_int bi = F->block_near_local[lbi];
            // Get indexes and sizes of corresponding block row and column
            STARSH_int i = F->block_near[2*bi];
            STARSH_int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            // Get pointers to data buffers
            double *D = M->near_D[lbi]->data;
            // Register data
            starpu_vector_data_register(work3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)D, (size_t)nrows*ncols, sizeof(*D));
            starpu_vector_data_register(B3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(temp_B+R->start[i]),
                    nrhs*(size_t)ldout-R->start[i], sizeof(*temp_B));
            starpu_vector_data_register(A3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(A+C->start[j]),
                    nrhs*(size_t)lda-C->start[j], sizeof(*A));
            // Multiply 2 dense matrices
            starpu_task_insert(&codelet, STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_VALUE, &nrhs, sizeof(nrhs),
                    STARPU_VALUE, &ncols, sizeof(ncols),
                    STARPU_VALUE, &alpha, sizeof(alpha),
                    STARPU_R, work3_handle[lbi],
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_R, A3_handle[lbi],
                    STARPU_VALUE, &lda, sizeof(lda),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_RW, B3_handle[lbi],
                    STARPU_VALUE, &ldout, sizeof(ldout),
                    0);
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
            //        nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
            //        out+R->start[i], ldout);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                starpu_vector_data_register(B4_handle+lbi, STARPU_MAIN_RAM,
                        (uintptr_t)(temp_B+C->start[j]),
                        nrhs*(size_t)ldout-C->start[j], sizeof(*temp_B));
                starpu_vector_data_register(A4_handle+lbi, STARPU_MAIN_RAM,
                        (uintptr_t)(A+R->start[i]),
                        nrhs*(size_t)lda-R->start[i], sizeof(*A));
                starpu_task_insert(&codelet, STARPU_VALUE, &T, sizeof(T),
                        STARPU_VALUE, &N, sizeof(N),
                        STARPU_VALUE, &ncols, sizeof(ncols),
                        STARPU_VALUE, &nrhs, sizeof(nrhs),
                        STARPU_VALUE, &nrows, sizeof(nrows),
                        STARPU_VALUE, &alpha, sizeof(alpha),
                        STARPU_R, work3_handle[lbi],
                        STARPU_VALUE, &nrows, sizeof(nrows),
                        STARPU_R, A4_handle[lbi],
                        STARPU_VALUE, &lda, sizeof(lda),
                        STARPU_VALUE, &one, sizeof(one),
                        STARPU_RW, B4_handle[lbi],
                        STARPU_VALUE, &ldout, sizeof(ldout),
                        0);
                //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncols,
                //        nrhs, nrows, alpha, D, nrows, A+R->start[i], lda,
                //        1.0, out+C->start[j], ldout);
                starpu_data_unregister_submit(A4_handle[lbi]);
                starpu_data_unregister_submit(B4_handle[lbi]);
            }
            starpu_data_unregister_submit(A3_handle[lbi]);
            starpu_data_unregister_submit(B3_handle[lbi]);
            starpu_data_unregister_submit(work3_handle[lbi]);
        }
    starpu_task_wait_for_all();
    // Since I keep result only on root node, following code is commented
    //for(int i = 0; i < nrhs; i++)
    //    MPI_Allreduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM,
    //            MPI_COMM_WORLD);
    for(int i = 0; i < nrhs; i++)
        MPI_Reduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
    free(temp_B);
    return STARSH_SUCCESS;
}



//int starsh_blrm__dmml_mpi_starpu_tlr(STARSH_blrm *matrix, int nrhs,
//        double alpha, double *A, int lda, double beta, double *B, int ldb)
//! Multiply tlr-matrix by dense matrix on MPI nodes (2D-block cycling).
/*! Performs `C=alpha*A*B+beta*C` with @ref STARSH_blrm `A` and dense matrices
 * `B` and `C`. All the integer types are int, since they are used in BLAS
 * calls. Block-wise low-rank matrix `A` is in TLR format.
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
    STARSH_int nblocks_far_local = F->nblocks_far_local;
    STARSH_int nblocks_near_local = F->nblocks_near_local;
    STARSH_int lbi;
    int T = CblasTrans, N = CblasNoTrans;
    double one = 1.0, zero = 0.0;
    char symm = F->symm;
    int maxrank = 0;
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
        if(maxrank < M->far_rank[lbi])
            maxrank = M->far_rank[lbi];
    STARSH_int maxnb = nrows/F->nbrows;
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int grid_nx = sqrt(mpi_size), grid_ny = grid_nx, grid_x, grid_y;
    if(grid_nx*grid_ny != mpi_size)
        STARSH_ERROR("MPI SIZE MUST BE SQUARE OF INTEGER!");
    grid_ny = mpi_size / grid_nx;
    grid_x = mpi_rank / grid_nx;
    grid_y = mpi_rank % grid_nx;
    MPI_Group mpi_leadingx_group, mpi_leadingy_group, mpi_world_group;
    MPI_Comm mpi_splitx, mpi_splity, mpi_leadingx, mpi_leadingy;
    MPI_Comm_group(MPI_COMM_WORLD, &mpi_world_group);
    int group_rank[grid_nx];
    for(int i = 0; i < grid_ny; i++)
        group_rank[i] = i;
    MPI_Group_incl(mpi_world_group, grid_ny, group_rank, &mpi_leadingy_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, mpi_leadingy_group, 0,
            &mpi_leadingy);
    for(int i = 0; i < grid_nx; i++)
        group_rank[i] = i*grid_ny;
    MPI_Group_incl(mpi_world_group, grid_nx, group_rank, &mpi_leadingx_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, mpi_leadingx_group, 0,
            &mpi_leadingx);
    MPI_Comm_split(MPI_COMM_WORLD, grid_x, mpi_rank, &mpi_splitx);
    MPI_Comm_split(MPI_COMM_WORLD, grid_y, mpi_rank, &mpi_splity);
    int mpi_leadingx_rank=-1, mpi_leadingx_size=-1;
    int mpi_leadingy_rank=-1, mpi_leadingy_size=-1;
    int mpi_splitx_rank, mpi_splitx_size;
    int mpi_splity_rank, mpi_splity_size;
    if(mpi_leadingx != MPI_COMM_NULL)
    {
        MPI_Comm_rank(mpi_leadingx, &mpi_leadingx_rank);
        MPI_Comm_size(mpi_leadingx, &mpi_leadingx_size);
    }
    if(mpi_leadingy != MPI_COMM_NULL)
    {
        MPI_Comm_rank(mpi_leadingy, &mpi_leadingy_rank);
        MPI_Comm_size(mpi_leadingy, &mpi_leadingy_size);
    }
    MPI_Comm_rank(mpi_splitx, &mpi_splitx_rank);
    MPI_Comm_size(mpi_splitx, &mpi_splitx_size);
    MPI_Comm_rank(mpi_splity, &mpi_splity_rank);
    MPI_Comm_size(mpi_splity, &mpi_splity_size);
    int grid_block_size = maxnb*grid_nx;
    int ld_temp_A = (F->nbcols+grid_nx-1-grid_x)/grid_nx*maxnb;
    double *temp_A;
    STARSH_MALLOC(temp_A, nrhs*(size_t)ld_temp_A);
    if(mpi_leadingx != MPI_COMM_NULL)
    {
        for(STARSH_int i = 0; i < F->nbcols/grid_nx; i++)
        {
            double *src = A+i*grid_block_size;
            double *recv = temp_A+i*maxnb;
            for(int j = 0; j < nrhs; j++)
            {
                MPI_Scatter(src+j*(size_t)lda, maxnb, MPI_DOUBLE,
                        recv+j*(size_t)ld_temp_A, maxnb, MPI_DOUBLE, 0,
                        mpi_leadingx);
            }
        }
        STARSH_int i = F->nbcols/grid_nx;
        int remain = F->nbcols-i*grid_nx;
        if(remain > 0)
        {
            double *src = A+i*(size_t)grid_block_size;
            double *recv = temp_A+i*(size_t)maxnb;
            if(mpi_rank == 0)
            {
                int sendcounts[grid_nx], displs[grid_nx];
                for(int j = 0; j < remain; j++)
                    sendcounts[j] = maxnb;
                for(int j = remain; j < grid_nx; j++)
                    sendcounts[j] = 0;
                displs[0] = 0;
                for(int j = 1; j < grid_nx; j++)
                    displs[j] = displs[j-1]+sendcounts[j-1];
                for(int j = 0; j < nrhs; j++)
                    MPI_Scatterv(src+j*(size_t)lda, sendcounts, displs,
                            MPI_DOUBLE, recv+j*(size_t)ld_temp_A, maxnb,
                            MPI_DOUBLE, 0, mpi_leadingx);
            }
            else
            {
                int recvcount = 0;
                if(grid_x < remain)
                    recvcount = maxnb;
                for(int j = 0; j < nrhs; j++)
                    MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                            recv+j*(size_t)ld_temp_A, recvcount, MPI_DOUBLE, 0,
                            mpi_leadingx);
            }
        }
    }
    MPI_Bcast(temp_A, nrhs*(size_t)ld_temp_A, MPI_DOUBLE, 0, mpi_splitx);
    double *temp_B;
    int ldout = (F->nbrows+grid_ny-1-grid_y)/grid_ny*maxnb;
    STARSH_MALLOC(temp_B, (size_t)nrhs*(size_t)ldout);
    starpu_data_handle_t U_handle[nblocks_far_local],
                         V_handle[nblocks_far_local],
                         A_handle[nblocks_far_local],
                         B_handle[nblocks_far_local],
                         work_handle[nblocks_far_local];
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
    // Setting temp_B=beta*B for master thread of root node and B=0 otherwise
    double *out = temp_B;
    for(size_t j = 0; j < nrhs*(size_t)ldout; j++)
        out[j] = 0.;
    if(beta != 0. && mpi_leadingy != MPI_COMM_NULL)
    {
        for(STARSH_int i = 0; i < F->nbrows/grid_ny; i++)
        {
            double *src = B+i*maxnb*grid_ny;
            double *recv = temp_B+i*maxnb;
            for(int j = 0; j < nrhs; j++)
                MPI_Scatter(src+j*(size_t)ldb, maxnb, MPI_DOUBLE,
                        recv+j*(size_t)ldout, maxnb, MPI_DOUBLE, 0,
                        mpi_leadingy);
        }
        for(int i = 0; i < ldout; i++)
            for(int j = 0; j < nrhs; j++)
                temp_B[j*(size_t)ldb+i] *= beta;
    }
    // Simple cycle over all far-field admissible blocks
    for(lbi = 0; lbi < nblocks_far_local; lbi++)
    {
        STARSH_int bi = F->block_far_local[lbi];
        // Get indexes of corresponding block row and block column
        STARSH_int i = F->block_far[2*bi];
        STARSH_int j = F->block_far[2*bi+1];
        // Get sizes and rank
        int nrows = R->size[i];
        int ncols = C->size[j];
        int rank = M->far_rank[lbi];
        if(rank == 0)
            continue;
        // Get pointers to data buffers
        double *U = M->far_U[lbi]->data, *V = M->far_V[lbi]->data;
        // Register data
        starpu_vector_data_register(U_handle+lbi, STARPU_MAIN_RAM,
                (uintptr_t)U, nrows*(size_t)rank, sizeof(*U));
        starpu_vector_data_register(V_handle+lbi, STARPU_MAIN_RAM,
                (uintptr_t)V, ncols*(size_t)rank, sizeof(*V));
        starpu_vector_data_register(work_handle+lbi, -1, 0, (size_t)nrhs*rank,
                sizeof(*U));
        starpu_vector_data_register(B_handle+lbi, STARPU_MAIN_RAM,
                (uintptr_t)(temp_B+(i/grid_ny)*maxnb),
                nrhs*(size_t)ldout-(i/grid_ny)*maxnb, sizeof(*temp_B));
        starpu_vector_data_register(A_handle+lbi, STARPU_MAIN_RAM,
                (uintptr_t)(temp_A+(j/grid_nx)*maxnb),
                nrhs*(size_t)ld_temp_A-(j/grid_nx)*maxnb, sizeof(*temp_A));
        // fake task to make workhandle initialized
        starpu_task_insert(&fake_codelet, STARPU_W, work_handle[lbi], 0);
        // Multiply low-rank matrix in U*V^T format by a dense matrix
        starpu_task_insert(&codelet, STARPU_VALUE, &T, sizeof(T),
                STARPU_VALUE, &N, sizeof(N),
                STARPU_VALUE, &rank, sizeof(rank),
                STARPU_VALUE, &nrhs, sizeof(nrhs),
                STARPU_VALUE, &ncols, sizeof(ncols),
                STARPU_VALUE, &one, sizeof(one),
                STARPU_R, V_handle[lbi],
                STARPU_VALUE, &ncols, sizeof(ncols),
                STARPU_R, A_handle[lbi],
                STARPU_VALUE, &ld_temp_A, sizeof(ld_temp_A),
                STARPU_VALUE, &zero, sizeof(zero),
                STARPU_RW, work_handle[lbi],
                STARPU_VALUE, &rank, sizeof(rank),
                0);
        //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
        //        ncols, 1.0, V, ncols, temp_A+(j/grid_nx)*maxnb, ld_temp_A,
        //        0.0,
        //        D, rank);
        starpu_task_insert(&codelet, STARPU_VALUE, &N, sizeof(N),
                STARPU_VALUE, &N, sizeof(N),
                STARPU_VALUE, &nrows, sizeof(nrows),
                STARPU_VALUE, &nrhs, sizeof(nrhs),
                STARPU_VALUE, &rank, sizeof(rank),
                STARPU_VALUE, &alpha, sizeof(alpha),
                STARPU_R, U_handle[lbi],
                STARPU_VALUE, &nrows, sizeof(nrows),
                STARPU_R, work_handle[lbi],
                STARPU_VALUE, &rank, sizeof(rank),
                STARPU_VALUE, &one, sizeof(one),
                STARPU_RW, B_handle[lbi],
                STARPU_VALUE, &ldout, sizeof(ldout),
                0);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, nrhs,
        //        rank, alpha, U, nrows, D, rank, 1.0, out+i/grid_ny*maxnb,
        //        ldout);
        starpu_data_unregister_submit(work_handle[lbi]);
        starpu_data_unregister_submit(U_handle[lbi]);
        starpu_data_unregister_submit(V_handle[lbi]);
        starpu_data_unregister_submit(A_handle[lbi]);
        starpu_data_unregister_submit(B_handle[lbi]);
    }
    starpu_task_wait_for_all();
    starpu_data_handle_t A3_handle[nblocks_near_local],
                         B3_handle[nblocks_near_local],
                         work3_handle[nblocks_near_local],
                         bi_handle[nblocks_near_local];
    STARSH_int bi_value[nblocks_near_local];
    if(M->onfly == 1)
        // Simple cycle over all near-field blocks
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
        {
            STARSH_int bi = F->block_near_local[lbi];
            // Get indexes and sizes of corresponding block row and column
            STARSH_int i = F->block_near[2*bi];
            STARSH_int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            // Register data
            bi_value[lbi] = bi;
            starpu_variable_data_register(bi_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(bi_value+lbi), sizeof(*bi_value));
            starpu_vector_data_register(work3_handle+lbi, -1, 0,
                    (size_t)nrows*ncols, sizeof(*A));
            starpu_vector_data_register(B3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(temp_B+(i/grid_ny)*maxnb),
                    nrhs*(size_t)ldout-(i/grid_ny)*maxnb, sizeof(*temp_B));
            starpu_vector_data_register(A3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(temp_A+(j/grid_nx)*maxnb),
                    nrhs*(size_t)ld_temp_A-(j/grid_nx)*maxnb, sizeof(*temp_A));
            // Fill temporary buffer with elements of corresponding block
            starpu_task_insert(&codelet_kernel, STARPU_VALUE, &F, sizeof(F),
                    STARPU_R, bi_handle[lbi], STARPU_W, work3_handle[lbi], 0);
            starpu_data_unregister_submit(bi_handle[lbi]);
            //kernel(nrows, ncols, R->pivot+R->start[i],
            //        C->pivot+C->start[j], RD, CD, D, nrows);
            // Multiply 2 dense matrices
            starpu_task_insert(&codelet, STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_VALUE, &nrhs, sizeof(nrhs),
                    STARPU_VALUE, &ncols, sizeof(ncols),
                    STARPU_VALUE, &alpha, sizeof(alpha),
                    STARPU_R, work3_handle[lbi],
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_R, A3_handle[lbi],
                    STARPU_VALUE, &ld_temp_A, sizeof(ld_temp_A),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_RW, B3_handle[lbi],
                    STARPU_VALUE, &ldout, sizeof(ldout),
                    0);
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
            //        nrhs, ncols, alpha, D, nrows,
            //        temp_A+(j/grid_nx)*(size_t)maxnb, ld_temp_A, 1.0,
            //        out+i/grid_ny*(size_t)maxnb, ldout);
            starpu_data_unregister_submit(A3_handle[lbi]);
            starpu_data_unregister_submit(B3_handle[lbi]);
            starpu_data_unregister_submit(work3_handle[lbi]);
        }
    else
        // Simple cycle over all near-field blocks
        for(lbi = 0; lbi < nblocks_near_local; lbi++)
        {
            STARSH_int bi = F->block_near_local[lbi];
            // Get indexes and sizes of corresponding block row and column
            STARSH_int i = F->block_near[2*bi];
            STARSH_int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            // Get pointers to data buffers
            double *D = M->near_D[lbi]->data;
            // Register data
            starpu_vector_data_register(work3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)D, (size_t)nrows*ncols, sizeof(*D));
            starpu_vector_data_register(B3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(temp_B+(i/grid_ny)*maxnb),
                    nrhs*(size_t)ldout-(i/grid_ny)*maxnb, sizeof(*temp_B));
            starpu_vector_data_register(A3_handle+lbi, STARPU_MAIN_RAM,
                    (uintptr_t)(temp_A+(j/grid_nx)*maxnb),
                    nrhs*(size_t)ld_temp_A-(j/grid_nx)*maxnb, sizeof(*temp_A));
            // Multiply 2 dense matrices
            starpu_task_insert(&codelet, STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &N, sizeof(N),
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_VALUE, &nrhs, sizeof(nrhs),
                    STARPU_VALUE, &ncols, sizeof(ncols),
                    STARPU_VALUE, &alpha, sizeof(alpha),
                    STARPU_R, work3_handle[lbi],
                    STARPU_VALUE, &nrows, sizeof(nrows),
                    STARPU_R, A3_handle[lbi],
                    STARPU_VALUE, &ld_temp_A, sizeof(ld_temp_A),
                    STARPU_VALUE, &one, sizeof(one),
                    STARPU_RW, B3_handle[lbi],
                    STARPU_VALUE, &ldout, sizeof(ldout),
                    0);
            //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
            //        nrhs, ncols, alpha, D, nrows,
            //        temp_A+(j/grid_nx)*(size_t)maxnb, ld_temp_A, 1.0,
            //        out+i/grid_ny*(size_t)maxnb, ldout);
            starpu_data_unregister_submit(A3_handle[lbi]);
            starpu_data_unregister_submit(B3_handle[lbi]);
            starpu_data_unregister_submit(work3_handle[lbi]);
        }
    starpu_task_wait_for_all();
    MPI_Barrier(MPI_COMM_WORLD);
    // Since I keep result only on root node, following code is commented
    //for(int i = 0; i < nrhs; i++)
    //    MPI_Allreduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM,
    //            MPI_COMM_WORLD);
    //for(int i = 0; i < nrhs; i++)
    //    MPI_Reduce(temp_B+i*ldout, B+i*ldb, ldout, MPI_DOUBLE, MPI_SUM, 0,
    //            MPI_COMM_WORLD);
    double *final_B = NULL;
    if(mpi_leadingy != MPI_COMM_NULL)
    {
        STARSH_MALLOC(final_B, nrhs*(size_t)ldout);
        for(size_t i = 0; i < nrhs*(size_t)ldout; i++)
            final_B[i] = 0.0;
    }
    MPI_Reduce(temp_B, final_B, nrhs*(size_t)ldout, MPI_DOUBLE, MPI_SUM, 0,
            mpi_splity);
    //STARSH_WARNING("REDUCE(%d): %f", mpi_rank, temp_B[0]);
    //if(mpi_splity_rank == 0)
    //    STARSH_WARNING("RESULT(%d): %f", mpi_rank, final_B[0]);
    if(mpi_leadingy != MPI_COMM_NULL)
    {
        for(STARSH_int i = 0; i < F->nbrows/grid_ny; i++)
        {
            double *src = final_B+i*(size_t)maxnb;
            double *recv = B+i*(size_t)maxnb*(size_t)grid_ny;
            for(int j = 0; j < nrhs; j++)
                MPI_Gather(src+j*(size_t)ldout, maxnb, MPI_DOUBLE,
                        recv+j*(size_t)ldb, maxnb, MPI_DOUBLE, 0,
                        mpi_leadingy);
        }
        STARSH_int i = F->nbrows/grid_ny;
        int remain = F->nbrows-i*grid_ny;
        if(remain > 0)
        {
            double *src = final_B+i*(size_t)maxnb;
            double *recv = B+i*(size_t)maxnb*(size_t)grid_ny;
            if(mpi_rank == 0)
            {
                int recvcounts[grid_ny], displs[grid_ny];
                for(int j = 0; j < remain; j++)
                    recvcounts[j] = maxnb;
                for(int j = remain; j < grid_ny; j++)
                    recvcounts[j] = 0;
                displs[0] = 0;
                for(int j = 1; j < grid_ny; j++)
                    displs[j] = displs[j-1]+recvcounts[j-1];
                for(int j = 0; j < nrhs; j++)
                    MPI_Gatherv(src+j*(size_t)ldout, maxnb, MPI_DOUBLE,
                            recv+j*(size_t)ldb, recvcounts, displs, MPI_DOUBLE,
                            0, mpi_leadingy);
            }
            else
            {
                int sendcount = 0;
                if(grid_y < remain)
                    sendcount = maxnb;
                for(int j = 0; j < nrhs; j++)
                    MPI_Gatherv(src+j*(size_t)ldout, sendcount, MPI_DOUBLE,
                            NULL, NULL, NULL, MPI_DOUBLE, 0, mpi_leadingy);
            }
        }
        MPI_Comm_free(&mpi_leadingy);
        free(final_B);
    }
    if(mpi_leadingx != MPI_COMM_NULL)
        MPI_Comm_free(&mpi_leadingx);
    MPI_Comm_free(&mpi_splitx);
    MPI_Comm_free(&mpi_splity);
    free(temp_A);
    free(temp_B);
    return STARSH_SUCCESS;
}

//! @endcond
