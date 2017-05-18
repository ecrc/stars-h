/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * @file backends/sequential/blrm/dca.c
 * @version 1.0.0.2
 * @author Aleksandr Mikhalev
 * @date 16 May 2017
 * */

#include "common.h"
#include "starsh.h"

int starsh_blrm__dca(STARSH_blrm *M, Array *A)
//! Convert double precision block low-rank matrix to dense Array
//! @ingroup blrm
/*! Memory for output array must be allocated prior calling this function.
 * */
{
    STARSH_blrf *F = M->format;
    STARSH_problem *P = F->problem;
    STARSH_cluster *RC = F->row_cluster, *CC = F->col_cluster;
    void *RD = RC->data, *CD = CC->data;
    int onfly = M->onfly, lda = A->shape[0];
    double *data = A->data;
    size_t bi;
    // At first restore far-field blocks
    for(bi = 0; bi < F->nblocks_far; bi++)
    {
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        double *U = M->far_U[bi]->data, *V = M->far_V[bi]->data;
        double *B = data+RC->start[i]+CC->start[j]*(size_t)lda;
        int nrows = RC->size[i], ncols = CC->size[j], rank = M->far_rank[bi];
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nrows, ncols,
                rank, 1.0, U, nrows, V, ncols, 0.0, B, lda);
        if(F->symm == 'S' && i != j)
        {
            double *B2 = data+CC->start[j]+RC->start[i]*(size_t)lda;
            for(int k = 0; k < ncols; k++)
                cblas_dcopy(nrows, B+k*lda, 1, B2+k, lda);
        }
    }
    // Restore near-field blocks
    for(bi = 0; bi < F->nblocks_near; bi++)
    {
        int i = F->block_near[2*bi];
        int j = F->block_near[2*bi+1];
        double *B = data+RC->start[i]+CC->start[j]*(size_t)lda;
        int nrows = RC->size[i], ncols = CC->size[j];
        double *D;
        if(onfly == 0)
            D = M->near_D[bi]->data;
        else
        {
            STARSH_MALLOC(D, (size_t)nrows*(size_t)ncols);
            P->kernel(nrows, ncols, RC->pivot+RC->start[i],
                    CC->pivot+CC->start[j], RD, CD, D);
        }
        for(int k = 0; k < ncols; k++)
            cblas_dcopy(nrows, D+k*nrows, 1, B+k*lda, 1);
        if(F->symm == 'S' && i != j)
        {
            double *B2 = data+CC->start[j]+RC->start[i]*(size_t)lda;
            for(int k = 0; k < ncols; k++)
                cblas_dcopy(nrows, D+k*nrows, 1, B2+k, lda);
        }
        if(onfly != 0)
            free(D);
    }
    return 0;
}
