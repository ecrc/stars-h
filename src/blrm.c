#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "stars.h"
#include "cblas.h"
#include "lapacke.h"


STARS_BLRM *STARS_BLRM_init(STARS_BLRF *blrf, int *far_rank, Array **far_U,
        Array **far_V, Array **far_D, int onfly, Array **near_D, void *U_alloc,
        void *V_alloc, void *D_alloc, char alloc_type)
// Init procedure for a non-nested block low-rank matrix
{
    STARS_BLRM *blrm = malloc(sizeof(*blrm));
    blrm->blrf = blrf;
    blrm->far_rank = far_rank;
    blrm->far_U = far_U;
    blrm->far_V = far_V;
    blrm->far_D = far_D;
    blrm->onfly = onfly;
    blrm->near_D = near_D;
    blrm->U_alloc = U_alloc;
    blrm->V_alloc = V_alloc;
    blrm->D_alloc = D_alloc;
    blrm->alloc_type = alloc_type;
    return blrm;
}

void STARS_BLRM_free(STARS_BLRM *blrm)
// Free memory of a non-nested block low-rank matrix
{
    if(blrm == NULL)
    {
        fprintf(stderr, "STARS_BLRM instance was NOT initialized\n");
        return;
    }
    STARS_BLRF *blrf = blrm->blrf;
    int i;
    free(blrm->far_rank);
    if(blrm->alloc_type == '1')
    {
        free(blrm->U_alloc);
        free(blrm->V_alloc);
        if(blrm->D_alloc != NULL)
            free(blrm->D_alloc);
    }
    else if(blrm->alloc_type == '2')
    {
        for(i = 0; i < blrf->nblocks_far; i++)
        {
            if(blrm->far_U[i] != NULL)
                Array_free(blrm->far_U[i]);
            if(blrm->far_V[i] != NULL)
                Array_free(blrm->far_V[i]);
            if(blrm->onfly == 0 && blrm->far_D[i] != NULL)
                Array_free(blrm->far_D[i]);
        }
        free(blrm->far_U);
        free(blrm->far_V);
        if(blrm->onfly == 0)
        {
            for(i = 0; i < blrf->nblocks_near; i++)
                if(blrm->near_D[i] != NULL)
                    Array_free(blrm->near_D[i]);
            free(blrm->far_D);
            free(blrm->near_D);
        }
    }
    else
    {
        fprintf(stderr, "Not supported allocation type\n");
        return;
    }
    free(blrm);
}

void STARS_BLRM_info(STARS_BLRM *blrm)
// Print short info on non-nested block low-rank matrix
{
    if(blrm == NULL)
    {
        fprintf(stderr, "STARS_BLRM instance is NOT initialized\n");
        return;
    }
    printf("<STARS_BLRM at %p, %d onfly, allocation type %d>\n",
            blrm, blrm->onfly, blrm->alloc_type);
}

void STARS_BLRM_error(STARS_BLRM *blrm)
// Measure error of approximation by non-nested block low-rank matrix
{
    STARS_BLRF *blrf = blrm->blrf;
    STARS_Problem *problem = blrf->problem;
    int bi, i, j, ndim = problem->ndim;
    if(ndim != 2)
    {
        fprintf(stderr, "Currently only scalar kernels are supported\n");
        return;
    }
    STARS_Cluster *row_cluster = blrf->row_cluster;
    STARS_Cluster *col_cluster = blrf->col_cluster;
    int nblocks_far = blrf->nblocks_far;
    int nblocks_near = blrf->nblocks_near;
    int nrowsi, ncolsj;
    double diff = 0., norm = 0., tmpnorm, tmpdiff, tmperr, maxerr = 0.;
    int *shape = (int *)malloc(ndim*sizeof(int));
    Array *block, *block2;
    char symm = blrf->symm;
    for(bi = 0; bi < nblocks_far; bi++)
    {
        i = blrf->block_far[2*bi];
        j = blrf->block_far[2*bi+1];
        if(i < j && symm == 'S')
            continue;
        nrowsi = row_cluster->size[i];
        ncolsj = col_cluster->size[j];
        shape[0] = nrowsi;
        shape[ndim-1] = ncolsj;
        block = Array_new(ndim, shape, problem->dtype, 'F');
        (problem->kernel)(nrowsi, ncolsj, row_cluster->pivot+
                row_cluster->start[i], col_cluster->pivot+
                col_cluster->start[j], problem->row_data, problem->col_data,
                block->buffer);
        tmpnorm = Array_norm(block);
        norm += tmpnorm*tmpnorm;
        if(i != j && symm == 'S')
            norm += tmpnorm*tmpnorm;
        if(blrm->far_U[bi] != NULL && blrm->far_V[bi] != NULL)
        {
            block2 = Array_dot(blrm->far_U[bi], blrm->far_V[bi]);
            tmpdiff = Array_diff(block, block2);
            Array_free(block2);
            diff += tmpdiff*tmpdiff;
            if(i != j && symm == 'S')
                diff += tmpdiff*tmpdiff;
            tmperr = tmpdiff/tmpnorm;
            if(tmperr > maxerr)
                maxerr = tmperr;
        }
        Array_free(block);
    }
    if(blrm->onfly == 0)
        for(bi = 0; bi < nblocks_near; bi++)
        {
            i = blrf->block_near[2*bi];
            j = blrf->block_near[2*bi+1];
            tmpnorm = Array_norm(blrm->near_D[bi]);
            norm += tmpnorm*tmpnorm;
            if(i != j && symm == 'S')
                norm += tmpnorm*tmpnorm;
        }
    else
        for(bi = 0; bi < nblocks_near; bi++)
        {
            i = blrf->block_near[2*bi];
            j = blrf->block_near[2*bi+1];
            nrowsi = row_cluster->size[i];
            ncolsj = col_cluster->size[j];
            shape[0] = nrowsi;
            shape[ndim-1] = ncolsj;
            block = Array_new(ndim, shape, problem->dtype, 'F');
            (problem->kernel)(nrowsi, ncolsj, row_cluster->pivot+
                    row_cluster->start[i], col_cluster->pivot+
                    col_cluster->start[j], problem->row_data, problem->col_data,
                    block->buffer);
            tmpnorm = Array_norm(block);
            norm += tmpnorm*tmpnorm;
            if(i != j && symm == 'S')
                norm += tmpnorm*tmpnorm;
            free(block);
        }
    printf("Relative error of approximation of full matrix: %e\n",
            sqrt(diff/norm));
    printf("Maximum relative error of per-block approximation: %e\n", maxerr);
    free(shape);
}

void STARS_BLRM_getblock(STARS_BLRM *blrm, int i, int j, int *shape, int *rank,
        void **U, void **V, void **D)
// Returns shape of block, its rank and low-rank factors or dense
// representation of a block
{
    STARS_BLRF *blrf = blrm->blrf;
    STARS_Problem *problem = blrf->problem;
    if(problem->ndim != 2)
    {
        fprintf(stderr, "Non-scalar kernels are not supported in STARS_BLRF_"
                "getblock\n");
        exit(1);
    }
    STARS_Cluster *row_cluster = blrf->row_cluster;
    STARS_Cluster *col_cluster = blrf->col_cluster;
    int nrows = row_cluster->size[i];
    int ncols = col_cluster->size[j];
    shape[0] = nrows;
    shape[1] = ncols;
    *rank = nrows < ncols ? nrows : ncols;
    *U = NULL;
    *V = NULL;
    *D = NULL;
    int bi = -1, k = blrf->brow_far_start[i];
    while(k < blrf->brow_far_start[i+1])
    {
        if(blrf->block_far[2*blrf->brow_far[k]+1] == j)
        {
            bi = k;
            break;
        }
        k++;
    }
    if(bi != -1)
    {
        *rank = blrm->far_rank[bi];
        *U = blrm->far_U[bi];
        *V = blrm->far_V[bi];
        *D = blrm->far_D[bi];
        if(*U == NULL && *V == NULL && *D == NULL)
            STARS_BLRF_getblock(blrf, i, j, shape, D);
        return;
    }
    k = blrf->brow_near_start[i];
    while(k < blrf->brow_near_start[i+1])
    {
        if(blrf->block_near[2*blrf->brow_near[k]+1] == j)
        {
            bi = k;
            break;
        }
        k++;
    }
    if(bi != -1)
    {
        *D = blrm->near_D[bi];
        if(*D == NULL)
            STARS_BLRF_getblock(blrf, i, j, shape, D);
        return;
    }
    printf("Required block (%d, %d) is not admissible!\n", i, j);
    STARS_BLRF_getblock(blrf, i, j, shape, D);
}

STARS_BLRM *STARS_blrf_tiled_compress_algebraic_svd(STARS_BLRF *blrf,
        int maxrank, double tol, int onfly)
// Private function of STARS-H
// Uses SVD to acquire rank of each block, compresses given matrix (given
// by block kernel, which returns submatrices) with relative accuracy tol
// or with given maximum rank (if maxrank <= 0, then tolerance is used)
{
    STARS_Problem *problem = blrf->problem;
    int bi, i, j, k, l, ndim = problem->ndim, mn, rank;
    if(ndim != 2)
    {
        fprintf(stderr, "Currently only scalar kernels are supported\n");
        return NULL;
    }
    char symm = blrf->symm;
    Array *block, *block2;
    Array *U, *S, *V;
    double *ptr, *ptrS, *ptrV;
    int nblocks_far = blrf->nblocks_far;
    int nblocks_near = blrf->nblocks_near;
    Array **far_U = malloc(nblocks_far*sizeof(Array *));
    Array **far_V = malloc(nblocks_far*sizeof(Array *));
    Array **far_D = NULL, **near_D = NULL;
    if(onfly == 0)
    {
        far_D = malloc(nblocks_far*sizeof(Array *));
        near_D = malloc(nblocks_near*sizeof(Array *));
    }
    else
    {
        far_D = NULL;
        near_D = NULL;
    }
    int *far_rank = malloc(nblocks_far*sizeof(int));
    STARS_Cluster *row_cluster = blrf->row_cluster;
    STARS_Cluster *col_cluster = blrf->col_cluster;
    int nrowsi, ncolsj;
    int *shape = malloc(ndim*sizeof(int));
    memcpy(shape, blrf->problem->shape, ndim*sizeof(int));
    for(bi = 0; bi < nblocks_far; bi++)
    // Cycle over every admissible block
    {
        i = blrf->block_far[2*bi];
        j = blrf->block_far[2*bi+1];
        nrowsi = row_cluster->size[i];
        ncolsj = col_cluster->size[j];
        shape[0] = nrowsi;
        shape[ndim-1] = ncolsj;
        mn = nrowsi > ncolsj ? ncolsj : nrowsi;
        far_U[bi] = NULL;
        far_V[bi] = NULL;
        far_D[bi] = NULL;
        far_rank[bi] = 0;
        if(i < j && symm == 'S')
            continue;
        block = Array_new(ndim, shape, problem->dtype, 'F');
        (problem->kernel)(nrowsi, ncolsj, row_cluster->pivot+
                row_cluster->start[i], col_cluster->pivot+
                col_cluster->start[j], problem->row_data, problem->col_data,
                block->buffer);
        block2 = Array_copy(block, 'N');
        //dmatrix_lr(rows, cols, block->buffer, tol, &rank, &U, &S, &V);
        Array_SVD(block, &U, &S, &V);
        rank = SVD_get_rank(S, tol, 'F');
        Array_free(block);
        if(rank < mn/2)
        // If block is low-rank
        {
            if(maxrank > 0)
                // If block is low-rank and maximum rank is upperbounded,
                // then rank should be equal to maxrank
                rank = maxrank;
            shape[0] = nrowsi;
            shape[1] = rank;
            far_U[bi] = Array_new(2, shape, 'd', 'F');
            shape[0] = rank;
            shape[1] = ncolsj;
            far_V[bi] = Array_new(2, shape, 'd', 'F');
            cblas_dcopy(rank*nrowsi, U->buffer, 1, far_U[bi]->buffer, 1);
            ptr = far_V[bi]->buffer;
            ptrS = S->buffer;
            ptrV = V->buffer;
            for(k = 0; k < ncolsj; k++)
                for(l = 0; l < rank; l++)
                {
                    ptr[k*rank+l] = ptrS[l]*ptrV[k*mn+l];
                }
            far_rank[bi] = rank;
            Array_free(block2);
        }
        else
        // If block is NOT low-rank
        {
            if(onfly == 0)
                far_D[bi] = block2;
            else
                Array_free(block2);
            far_rank[bi] = mn;
        }
        Array_free(U);
        Array_free(S);
        Array_free(V);
    }
    if(onfly == 0)
        for(bi = 0; bi < nblocks_near; bi++)
        {
            i = blrf->block_near[2*bi];
            j = blrf->block_near[2*bi+1];
            nrowsi = row_cluster->size[i];
            ncolsj = col_cluster->size[j];
            shape[0] = nrowsi;
            shape[ndim-1] = ncolsj;
            mn = nrowsi > ncolsj ? ncolsj : nrowsi;
            block = Array_new(ndim, shape, problem->dtype, 'F');
            (problem->kernel)(nrowsi, ncolsj, row_cluster->pivot+
                    row_cluster->start[i], col_cluster->pivot+
                    col_cluster->start[j], problem->row_data,
                    problem->col_data, block->buffer);
            near_D[bi] = block;
        }
    free(shape);
    return STARS_BLRM_init(blrf, far_rank, far_U, far_V, far_D, onfly, near_D,
            NULL, NULL, NULL, '2');
}

STARS_BLRM *STARS_blrf_tiled_compress_algebraic_svd_ompfor(STARS_BLRF *blrf,
        int maxrank, double tol, int onfly)
// Private function of STARS-H
// Uses SVD to acquire rank of each block, compresses given matrix (given
// by block kernel, which returns submatrices) with relative accuracy tol
// or with given maximum rank (if maxrank <= 0, then tolerance is used)
{
    STARS_Problem *problem = blrf->problem;
    int bi, i = 0, j, k, l, ndim = problem->ndim, mn, rank;
    if(ndim != 2)
    {
        fprintf(stderr, "Currently only scalar kernels are supported\n");
        return NULL;
    }
    int nblocks_far = blrf->nblocks_far;
    for(bi = 0; bi < nblocks_far; bi++)
    {
        ;
    }
    return NULL;
}
