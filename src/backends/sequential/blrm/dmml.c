#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "stars.h"
#include "misc.h"

int starsh_blrm__dmml(STARS_BLRM *M, int nrhs, double *A, int lda,
        double *B, int ldb)
// Double precision Multiply by dense Matrix, blr-matrix is on Left side
{
    STARS_BLRF *F = M->blrf;
    STARS_Problem *P = F->problem;
    block_kernel kernel = P->kernel;
    // Shorcuts to information about clusters
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    void *RD = R->data, *CD = C->data;
    // Number of far-field and near-field blocks
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near, bi;
    char symm = F->symm;
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
        STARS_MALLOC(D, nrhs*(size_t)rank);
        // Multiply low-rank matrix in U*V^T format by a dense matrix
        cblas_dgemm(LAPACK_COL_MAJOR, CblasTrans, CblasNoTrans, rank, nrhs,
                ncols, 1.0, V, ncols, A+C->start[j], lda, 0.0, D, rank);
        cblas_dgemm(LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, nrows, nrhs,
                rank, 1.0, U, nrows, D, rank, 1.0, B+R->start[i], ldb);
        if(i != j && symm == 'S')
        {
            // Multiply low-rank matrix in V*U^T format by a dense matrix
            // U and V are simply swapped in case of symmetric block
            cblas_dgemm(LAPACK_COL_MAJOR, CblasTrans, CblasNoTrans, rank, nrhs,
                    nrows, 1.0, U, nrows, A+R->start[i], lda, 0.0, D, rank);
            cblas_dgemm(LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, ncols,
                    nrhs, rank, 1.0, V, ncols, D, rank, 1.0,
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
            STARS_MALLOC(D, (size_t)nrows*(size_t)ncols);
            // Fill temporary buffer with elements of corresponding block
            kernel(nrows, ncols, R->pivot+R->start[i], C->pivot+C->start[j],
                    RD, CD, D);
            // Multiply 2 dense matrices
            cblas_dgemm(LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, 1.0, D, nrows, A+C->start[j], lda, 1.0,
                    B+R->start[i], ldb);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                cblas_dgemm(LAPACK_COL_MAJOR, CblasTrans, CblasNoTrans,
                        ncols, nrhs, nrows, 1.0, D, nrows, A+R->start[i], lda,
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
            cblas_dgemm(LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, 1.0, D, nrows, A+C->start[j], lda, 1.0,
                    B+R->start[i], ldb);
            if(i != j && symm == 'S')
            {
                // Repeat in case of symmetric matrix
                cblas_dgemm(LAPACK_COL_MAJOR, CblasTrans, CblasNoTrans,
                        ncols, nrhs, nrows, 1.0, D, nrows, A+R->start[i], lda,
                        1.0, B+C->start[j], ldb);
            }
        }
    return 0;
}
