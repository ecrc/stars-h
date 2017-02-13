#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "starsh.h"

int starsh_blrm__dmml(STARSH_blrm *M, int nrhs, double alpha, double *A,
        int lda, double beta, double *B, int ldb)
//! Double precision Multiply by dense Matrix, blr-matrix is on Left side.
/*! Performs `B=alpha*M*A+beta*B` */
{
    STARSH_blrf *F = M->format;
    STARSH_problem *P = F->problem;
    STARSH_kernel kernel = P->kernel;
    int nrows = P->shape[0];
    int ncols = P->shape[P->ndim-1];
    // Shorcuts to information about clusters
    STARSH_cluster *R = F->row_cluster, *C = F->col_cluster;
    void *RD = R->data, *CD = C->data;
    // Number of far-field and near-field blocks
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near, bi;
    char symm = F->symm;
    // Setting B = beta*B
    if(beta == 0.)
        for(int i = 0; i < nrhs; i++)
            for(int j = 0; j < nrows; j++)
                B[i*ldb+j] = 0.;
    else
        for(int i = 0; i < nrhs; i++)
            for(int j = 0; j < nrows; j++)
                B[i*ldb+j] *= beta;
    // Simple cycle over all far-field admissible blocks
    for(bi = 0; bi < nblocks_far; bi++)
    {
        // Get indexes of corresponding block row and block column
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        // Get sizes and rank
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
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            // Get pointers to data buffers
            double *D;
            // Allocate temporary buffer
            STARSH_MALLOC(D, (size_t)nrows*(size_t)ncols);
            // Fill temporary buffer with elements of corresponding block
            kernel(nrows, ncols, R->pivot+R->start[i], C->pivot+C->start[j],
                    RD, CD, D);
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
            free(D);
        }
    if(M->onfly == 0)
        // Simple cycle over all near-field blocks
        for(bi = 0; bi < nblocks_near; bi++)
        {
            // Get indexes and sizes of corresponding block row and column
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
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
