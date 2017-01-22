#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "stars.h"
#include "misc.h"

int starsh_blrm__dca(STARS_BLRM *M, Array *A)
// Double precision Convert to Array
//
// Create dense version of approximation
{
    STARS_BLRF *F = M->blrf;
    STARS_Problem *P = F->problem;
    STARS_Cluster *RC = F->row_cluster, *CC = F->col_cluster;
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
        if(F->symm == 'S')
        {
            double *B2 = data+CC->start[j]+RC->start[i]*(size_t)lda;
            for(int k = 0; k < ncols; k++)
                cblas_dcopy(nrows, B+k*lda, 1, B2+k, lda);
        }
    }
    // Restore near-field blocks
    for(bi = 0; bi < F->nblocks_near; bi++)
    {
        //printf("NEAR %zu\n", bi);
        int i = F->block_near[2*bi];
        int j = F->block_near[2*bi+1];
        double *B = data+RC->start[i]+CC->start[j]*(size_t)lda;
        int nrows = RC->size[i], ncols = CC->size[j];
        double *D;
        if(onfly == 0)
            D = M->near_D[bi]->data;
        else
        {
            STARS_MALLOC(D, (size_t)nrows*(size_t)ncols);
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
