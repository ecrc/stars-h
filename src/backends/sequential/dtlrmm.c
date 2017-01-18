#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include "stars.h"
#include "misc.h"

int dtlrmm_l(STARS_BLRM *M, Array *A, Array *B)
{
    STARS_BLRF *F = M->blrf;
    STARS_Problem *P = F->problem;
    block_kernel kernel = P->kernel;
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    void *RD = R->data, *CD = C->data;
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near, bi;
    size_t nrhs = A->size/A->shape[0];
    char symm = F->symm;
    for(bi = 0; bi < nblocks_far; bi++)
    {
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        int nrows = R->size[i];
        int ncols = C->size[j];
        int rank = M->far_rank[bi];
        double *D, *U = M->far_U[bi]->data, *V = M->far_V[bi]->data;
        double *rhs = (double *)A->data+C->start[j];
        double *result = (double *)B->data+R->start[i];
        STARS_MALLOC(D, nrhs*(size_t)rank);
        cblas_dgemm(LAPACK_COL_MAJOR, CblasTrans, CblasNoTrans, rank, nrhs,
                ncols, 1.0, V, ncols, rhs, A->shape[0], 0.0, D, rank);
        cblas_dgemm(LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, nrows, nrhs,
                rank, 1.0, U, nrows, D, rank, 1.0, result, B->shape[0]);
        if(i != j && symm == 'S')
        {
            rhs = (double *)A->data+R->start[i];
            result = (double *)B->data+C->start[j];
            cblas_dgemm(LAPACK_COL_MAJOR, CblasTrans, CblasNoTrans, rank, nrhs,
                    nrows, 1.0, U, nrows, rhs, A->shape[0], 0.0, D, rank);
            cblas_dgemm(LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, ncols,
                    nrhs, rank, 1.0, V, ncols, D, rank, 1.0, result, B->shape[0]);
        }
        free(D);
    }
    if(M->onfly == 1)
        for(bi = 0; bi < nblocks_near; bi++)
        {
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            double *D;
            double *rhs = (double *)A->data+C->start[j];
            double *result = (double *)B->data+R->start[i];
            STARS_MALLOC(D, (size_t)nrows*(size_t)ncols);
            kernel(nrows, ncols, R->pivot+R->start[i], C->pivot+C->start[j],
                    RD, CD, D);
            cblas_dgemm(LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, 1.0, D, nrows, rhs, A->shape[0], 1.0,
                    result, B->shape[0]);
            if(i != j && symm == 'S')
            {
                rhs = (double *)A->data+R->start[i];
                result = (double *)B->data+C->start[j];
                cblas_dgemm(LAPACK_COL_MAJOR, CblasTrans, CblasNoTrans,
                        ncols, nrhs, nrows, 1.0, D, nrows, rhs, A->shape[0],
                        1.0, result, B->shape[0]);
            }
            free(D);
        }
    if(M->onfly == 0)
        for(bi = 0; bi < nblocks_near; bi++)
        {
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            int nrows = R->size[i];
            int ncols = C->size[j];
            double *D = M->near_D[bi]->data;
            double *rhs = (double *)A->data+C->start[j];
            double *result = (double *)B->data+R->start[i];
            cblas_dgemm(LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, nrows,
                    nrhs, ncols, 1.0, D, nrows, rhs, A->shape[0], 1.0,
                    result, B->shape[0]);
            if(i != j && symm == 'S')
            {
                rhs = (double *)A->data+R->start[i];
                result = (double *)B->data+C->start[j];
                cblas_dgemm(LAPACK_COL_MAJOR, CblasTrans, CblasNoTrans,
                        ncols, nrhs, nrows, 1.0, D, nrows, rhs, A->shape[0],
                        1.0, result, B->shape[0]);
            }
        }
    return 0;
}
