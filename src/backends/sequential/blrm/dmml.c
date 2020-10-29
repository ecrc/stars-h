/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/sequential/blrm/dmml.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "starsh.h"
#include "common.h"

int starsh_blrm__dmml(STARSH_blrm *matrix, int nrhs, double alpha, double *A,
        int lda, double beta, double *B, int ldb)
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
        double *D, *U = M->far_U[bi]->data, *V = M->far_V[bi]->data;
        // Allocate temporary buffer
        STARSH_MALLOC(D, nrhs*(size_t)rank);
        // Multiply low-rank matrix in U*V^T format by a dense matrix
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                ncols, 1.0, V, ncols, A+C->start[j], lda, 0.0, D, rank);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, nrhs,
                rank, alpha, U, nrows, D, rank, 1.0, B+R->start[i], ldb);
        if(i != j && symm == 'S')
        {
            // Multiply low-rank matrix in V*U^T format by a dense matrix
            // U and V are simply swapped in case of symmetric block
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                    nrows, 1.0, U, nrows, A+R->start[i], lda, 0.0, D, rank);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ncols,
                    nrhs, rank, alpha, V, ncols, D, rank, 1.0,
                    B+C->start[j], ldb);
        }
        free(D);
    }
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
            // Get pointers to data buffers
            double *D;
            // Allocate temporary buffer
            STARSH_MALLOC(D, (size_t)nrows*ncols);
            // Fill temporary buffer with elements of corresponding block
            kernel(nrows, ncols, R->pivot+R->start[i], C->pivot+C->start[j],
                    RD, CD, D, nrows);
            // Multiply 2 dense matrices
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                    B+R->start[i], ldb);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        ncols, nrhs, nrows, alpha, D, nrows, A+R->start[i],
                        lda, 1.0, B+C->start[j], ldb);
            }
            free(D);
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
            // Multiply 2 dense matrices
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                    B+R->start[i], ldb);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        ncols, nrhs, nrows, alpha, D, nrows, A+R->start[i], lda,
                        1.0, B+C->start[j], ldb);
            }
        }
    return 0;
}
