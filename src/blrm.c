#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "stars.h"
#include "cblas.h"
#include "lapacke.h"


STARS_BLRM *STARS_BLRM_init(STARS_BLRF *blrf, int nblocks, int *brank,
        Array **U, Array **V, Array **D, void *U_alloc, void *V_alloc,
        void *D_alloc, char alloc_type)
{
    STARS_BLRM *blrm = malloc(sizeof(*blrm));
    blrm->blrf = blrf;
    blrm->nblocks = nblocks;
    blrm->brank = brank;
    blrm->U = U;
    blrm->V = V;
    blrm->D = D;
    blrm->U_alloc = U_alloc;
    blrm->V_alloc = V_alloc;
    blrm->D_alloc = D_alloc;
    blrm->alloc_type = alloc_type;
    return blrm;
}

void STARS_BLRM_free(STARS_BLRM *blrm)
{
    if(blrm->alloc_type == '1')
    {
        free(blrm->U_alloc);
        free(blrm->V_alloc);
        free(blrm->D_alloc);
    }
    else if(blrm->alloc_type == '2')
    {
        for(int i = 0; i < blrm->nblocks; i++)
        {
            if(blrm->U[i] != NULL)
                Array_free(blrm->U[i]);
            if(blrm->V[i] != NULL)
                Array_free(blrm->V[i]);
            if(blrm->D[i] != NULL)
                Array_free(blrm->D[i]);
        }
    }
    else
    {
        fprintf(stderr, "Not supported allocation type\n");
        return;
    }
    free(blrm->brank);
    free(blrm);
}

void STARS_BLRM_info(STARS_BLRM *blrm)
{
    if(blrm == NULL)
    {
        fprintf(stderr, "STARS_BLRM instance is NOT initialized\n");
        return;
    }
    printf("<STARS_BLRM at %p, %d admissible blocks, allocation type %d>\n",
            blrm, blrm->nblocks, blrm->alloc_type);
}

STARS_BLRM *STARS_blrf_tiled_compress_algebraic_svd(STARS_BLRF *blrf,
        int maxrank, double tol)
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
    char symm = blrf->symm;
    Array *block, *block2;
    Array *U, *S, *V;
    double *ptr, *ptrS, *ptrV;
    //double norm;
    int nblocks = blrf->admissible_nblocks;
    int nbrows = blrf->nbrows;
    //int nbcols = blrf->nbcols;
    Array **blrmU = malloc(nblocks*sizeof(Array *));
    Array **blrmV = malloc(nblocks*sizeof(Array *));
    Array **blrmD = malloc(nblocks*sizeof(Array *));
    int *brank = malloc(nblocks*sizeof(int));
    STARS_Cluster *row_cluster = blrf->row_cluster;
    STARS_Cluster *col_cluster = blrf->col_cluster;
    int nrowsi, ncolsj;
    int *shape = malloc(ndim*sizeof(int));
    memcpy(shape, blrf->problem->shape, ndim*sizeof(int));
    for(bi = 0; bi < nblocks; bi++)
    // Cycle over every admissible block
    {
        if(i+1 < nbrows)
            if(blrf->ibrow_admissible_start[i+1] == bi)
                i += 1;
        j = blrf->ibrow_admissible[bi];
        nrowsi = row_cluster->size[i];
        ncolsj = col_cluster->size[j];
        shape[0] = nrowsi;
        shape[ndim-1] = ncolsj;
        mn = nrowsi > ncolsj ? ncolsj : nrowsi;
        if(i < j && symm == 'S')
        {
            blrmU[bi] = NULL;
            blrmV[bi] = NULL;
            blrmD[bi] = NULL;
            brank[bi] = mn;
            continue;
        }
        block = Array_new(ndim, shape, blrf->problem->dtype, 'F');
        (problem->kernel)(nrowsi, ncolsj, row_cluster->pivot+
                row_cluster->start[i], col_cluster->pivot+
                col_cluster->start[j], problem->row_data, problem->col_data,
                block->buffer);
        if(blrf->ibrow_admissible_status[bi] == STARS_Dense)
        {
            blrmU[bi] = NULL;
            blrmV[bi] = NULL;
            blrmD[bi] = block;
        }
        else// blrf->ibrow_admissible_status[bi] != STARS_Dense
        {
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
                blrmD[bi] = NULL;
                shape[0] = nrowsi;
                shape[1] = rank;
                blrmU[bi] = Array_new(2, shape, 'd', 'F');
                shape[0] = rank;
                shape[1] = ncolsj;
                blrmV[bi] = Array_new(2, shape, 'd', 'F');
                cblas_dcopy(rank*nrowsi, U->buffer, 1, blrmU[bi]->buffer, 1);
                ptr = blrmV[bi]->buffer;
                ptrS = S->buffer;
                ptrV = V->buffer;
                for(k = 0; k < ncolsj; k++)
                    for(l = 0; l < rank; l++)
                    {
                        ptr[k*rank+l] = ptrS[l]*ptrV[k*mn+l];
                    }
                brank[bi] = rank;
                Array_free(block2);
            }
            else
                // If block is NOT low-rank
            {
                blrmU[bi] = NULL;
                blrmV[bi] = NULL;
                blrmD[bi] = block2;
                brank[bi] = mn;
            }
            Array_free(U);
            Array_free(S);
            Array_free(V);
        }
    }
    free(shape);
    return STARS_BLRM_init(blrf, nblocks, brank, blrmU, blrmV, blrmD,
            NULL, NULL, NULL, '2');
}

void STARS_BLRM_error(STARS_BLRM *blrm)
{
    STARS_BLRF *blrf = blrm->blrf;
    STARS_Problem *problem = blrf->problem;
    int bi, i = 0, j, ndim = problem->ndim;
    if(ndim != 2)
    {
        fprintf(stderr, "Currently only scalar kernels are supported\n");
        return;
    }
    STARS_Cluster *row_cluster = blrf->row_cluster;
    STARS_Cluster *col_cluster = blrf->col_cluster;
    int nblocks = blrm->nblocks;
    int nbrows = blrf->nbrows;
    //int nbcols = blrf->nbcols;
    int nrowsi, ncolsj;
    double diff = 0., norm = 0., tmpnorm, tmpdiff, tmperr, maxerr = 0.;
    int *shape = (int *)malloc(ndim*sizeof(int));
    Array *block, *block2;
    char symm = blrf->symm;
    for(bi = 0; bi < nblocks; bi++)
    {
        if(i+1 < nbrows)
            if(blrf->ibrow_admissible_start[i+1] == bi)
                i += 1;
        j = blrf->ibrow_admissible[bi];
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
        if(blrm->D[bi] == NULL)
        {
            block2 = Array_dot(blrm->U[bi], blrm->V[bi]);
            tmpdiff = Array_diff(block, block2);
            Array_free(block2);
            diff += tmpdiff*tmpdiff;
            if(i != j && symm == 'S')
                diff += tmpdiff*tmpdiff;
            tmperr = tmpdiff/tmpnorm;
            if(tmperr > maxerr)
                maxerr = tmperr;
        }
        if(blrm->D[bi] != NULL)
        {
            block2 = blrm->D[bi];
            tmpdiff = Array_diff(block, block2);
            diff += tmpdiff*tmpdiff;
        }
        Array_free(block);
    }
    printf("Relative error of approximation of full matrix: %e\n",
            sqrt(diff/norm));
    printf("Maximum relative error of per-block approximation: %e\n", maxerr);
    free(shape);
}
