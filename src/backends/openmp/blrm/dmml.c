#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <omp.h>
#include "starsh.h"

int starsh_blrm__dmml_omp(STARSH_blrm *M, int nrhs, double alpha, double *A,
        int lda, double beta, double *B, int ldb)
//! Double precision Multiply by dense Matrix, blr-matrix is on Left side.
/*! Performs `B=alpha*M*A+beta*B` using OpenMP */
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
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < nrows; i++)
            for(int j = 0; j < nrhs; j++)
                B[j*ldb+i] = 0.;
    else
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < nrows; i++)
            for(int j = 0; j < nrhs; j++)
                B[j*ldb+i] *= beta;
    // Simple cycle over all far-field admissible blocks
    #pragma omp parallel
    #pragma omp single nowait
    {
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
            double *U = M->far_U[bi]->data, *V = M->far_V[bi]->data;
            int info = 0;
            if(i == j || symm == 'N')
            #pragma omp task depend(inout:B[R->start[i]]) \
                    //firstprivate(i, j, nrows, ncols, rank, U, V, info)
            {
                double *D;
                // Allocate temporary buffer
                STARSH_PMALLOC(D, nrhs*(size_t)rank, info);
                // Multiply low-rank matrix in U*V^T format by a dense matrix
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                        ncols, 1.0, V, ncols, A+C->start[j], lda, 0.0, D, rank);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, nrhs,
                        rank, alpha, U, nrows, D, rank, 1.0, B+R->start[i], ldb);
                free(D);
            }
            else
            #pragma omp task depend(inout:B[R->start[i]], B[C->start[j]]) \
                    //firstprivate(i, j, nrows, ncols, rank, U, V, info)
            {
                double *D;
                // Allocate temporary buffer
                STARSH_PMALLOC(D, nrhs*(size_t)rank, info);
                // Multiply low-rank matrix in U*V^T format by a dense matrix
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                        ncols, 1.0, V, ncols, A+C->start[j], lda, 0.0, D, rank);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, nrhs,
                        rank, alpha, U, nrows, D, rank, 1.0, B+R->start[i], ldb);
                // Multiply low-rank matrix in V*U^T format by a dense matrix
                // U and V are simply swapped in case of symmetric block
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nrhs,
                        nrows, 1.0, U, nrows, A+R->start[i], lda, 0.0, D, rank);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ncols,
                        nrhs, rank, alpha, V, ncols, D, rank, 1.0,
                        B+C->start[j], ldb);
                free(D);
            }
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
                // Allocate temporary buffer
                int info = 0;
                if(i == j || symm == 'N')
                #pragma omp task depend(inout:B[R->start[i]]) \
                        //firstprivate(i, j, nrows, ncols, info)
                {
                    double *D;
                    STARSH_PMALLOC(D, (size_t)nrows*(size_t)ncols, info);
                    // Fill temporary buffer with elements of corresponding block
                    kernel(nrows, ncols, R->pivot+R->start[i],
                            C->pivot+C->start[j], RD, CD, D);
                    // Multiply 2 dense matrices
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                            nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                            B+R->start[i], ldb);
                    free(D);
                }
                else
                #pragma omp task depend(inout:B[R->start[i]], B[C->start[j]]) \
                        //firstprivate(i, j, nrows, ncols, info)
                {
                    double *D;
                    STARSH_PMALLOC(D, (size_t)nrows*(size_t)ncols, info);
                    // Fill temporary buffer with elements of corresponding block
                    kernel(nrows, ncols, R->pivot+R->start[i],
                            C->pivot+C->start[j], RD, CD, D);
                    // Multiply 2 dense matrices
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                            nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                            B+R->start[i], ldb);
                    // Repeat in case of symmetric matrix
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            ncols, nrhs, nrows, alpha, D, nrows, A+R->start[i], lda,
                            1.0, B+C->start[j], ldb);
                    free(D);
                }
            }
        else
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
                if(i == j || symm == 'N')
                #pragma omp task depend(inout:B[R->start[i]]) \
                        //firstprivate(i, j, nrows, ncols, D)
                {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                            nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                            B+R->start[i], ldb);
                }
                else
                #pragma omp task depend(inout:B[R->start[i]], B[C->start[j]]) \
                        //firstprivate(i, j, nrows, ncols, D)
                {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows,
                            nrhs, ncols, alpha, D, nrows, A+C->start[j], lda, 1.0,
                            B+R->start[i], ldb);
                    // Repeat in case of symmetric matrix
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            ncols, nrhs, nrows, alpha, D, nrows, A+R->start[i], lda,
                            1.0, B+C->start[j], ldb);
                }
            }
    }
    return 0;
}
