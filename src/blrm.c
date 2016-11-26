#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "stars.h"
#include "cblas.h"
#include "lapacke.h"
#include "misc.h"


int STARS_BLRM_new(STARS_BLRM **M, STARS_BLRF *F, int *far_rank, Array **far_U,
        Array **far_V, int onfly, Array **near_D, void *alloc_U, void *alloc_V,
        void *alloc_D, char alloc_type)
// Init procedure for a non-nested block low-rank matrix
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(F == NULL)
    {
        STARS_ERROR("invalid value of `F`");
        return 1;
    }
    if(far_rank == NULL && F->nblocks_far > 0)
    {
        STARS_ERROR("invalid value of `far_rank`");
        return 1;
    }
    if(far_U == NULL && F->nblocks_far > 0)
    {
        STARS_ERROR("invalid value of `far_U`");
        return 1;
    }
    if(far_V == NULL && F->nblocks_far > 0)
    {
        STARS_ERROR("invalid value of `far_V`");
        return 1;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARS_ERROR("invalid value of `onfly`");
        return 1;
    }
    if(near_D == NULL && F->nblocks_near > 0 && onfly == 0)
    {
        STARS_ERROR("invalid value of `near_D`");
        return 1;
    }
    if(alloc_type != '1' && alloc_type != '2')
    {
        STARS_ERROR("invalid value of `alloc_type`");
        return 1;
    }
    if(alloc_U == NULL && alloc_type == '1')
    {
        STARS_ERROR("invalid value of `alloc_U`");
        return 1;
    }
    if(alloc_V == NULL && alloc_type == '1')
    {
        STARS_ERROR("invalid value of `alloc_V`");
        return 1;
    }
    if(alloc_D == NULL && alloc_type == '1' && onfly == 0)
    {
        STARS_ERROR("invalid value of `alloc_D`");
        return 1;
    }
    STARS_MALLOC(*M, 1);
    STARS_BLRM *M2 = *M;
    M2->blrf = F;
    M2->far_rank = far_rank;
    M2->far_U = far_U;
    M2->far_V = far_V;
    M2->onfly = onfly;
    M2->near_D = near_D;
    M2->alloc_U = alloc_U;
    M2->alloc_V = alloc_V;
    M2->alloc_D = alloc_D;
    M2->alloc_type = alloc_type;
    return 0;
}

int STARS_BLRM_free(STARS_BLRM *M)
// Free memory of a non-nested block low-rank matrix
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    STARS_BLRF *F = M->blrf;
    size_t bi;
    int info;
    if(F->nblocks_far > 0)
    {
        if(M->alloc_type == '1')
        {
            free(M->alloc_U);
            free(M->alloc_V);
            for(bi = 0; bi < F->nblocks_far; bi++)
            {
                M->far_U[bi]->data = NULL;
                info = Array_free(M->far_U[bi]);
                if(info != 0)
                    return info;
                M->far_V[bi]->data = NULL;
                info = Array_free(M->far_V[bi]);
                if(info != 0)
                    return info;
            }
        }
        else// M->alloc_type == '2'
        {
            for(bi = 0; bi < F->nblocks_far; bi++)
            {
                info = Array_free(M->far_U[bi]);
                if(info != 0)
                    return info;
                info = Array_free(M->far_V[bi]);
                if(info != 0)
                    return info;
            }
        }
        free(M->far_rank);
        free(M->far_U);
        free(M->far_V);
    }
    if(F->nblocks_near > 0 && M->onfly == 0)
    {
        if(M->alloc_type == '1')
        {
            free(M->alloc_D);
            for(bi = 0; bi < F->nblocks_near; bi++)
            {
                M->near_D[bi]->data = NULL;
                info = Array_free(M->near_D[bi]);
                if(info != 0)
                    return info;
            }
        }
        else// M->alloc_type == '2'
        {
            for(bi = 0; bi < F->nblocks_near; bi++)
            {
                info = Array_free(M->near_D[bi]);
                if(info != 0)
                    return info;
            }
        }
        free(M->near_D);
    }
    free(M);
    return 0;
}

int STARS_BLRM_info(STARS_BLRM *M)
// Print short info on non-nested block low-rank matrix
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    printf("<STARS_BLRM at %p, %d onfly, allocation type '%c'>\n",
            M, M->onfly, M->alloc_type);
    return 0;
}

int STARS_BLRM_error(STARS_BLRM *M)
// Measure error of approximation by non-nested block low-rank matrix
{
    double tmp_time = omp_get_wtime(), tmp_time2;
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    STARS_BLRF *F = M->blrf;
    STARS_Problem *P = F->problem;
    int ndim = P->ndim;
    if(ndim != 2)
    {
        STARS_ERROR("only scalar kernels are supported");
        return 1;
    }
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near, bi;
    double diff = 0., norm = 0., maxerr = 0.;
    char symm = F->symm;
    int info = 0;
    volatile int abort = 0;
    #pragma omp parallel for
    for(bi = 0; bi < nblocks_far; bi++)
    {
        if(abort != 0)
            continue;
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        int nrowsi = R->size[i];
        int ncolsj = C->size[j];
        int shape[2] = {nrowsi, ncolsj};
        Array *A, *A2;
        int info = Array_new(&A, 2, shape, P->dtype, 'F');
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                C->pivot+C->start[j], P->row_data, P->col_data, A->data);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        double tmpnorm, tmpdiff;
        info = Array_norm(A, &tmpnorm);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        #pragma omp critical
        {
            norm += tmpnorm*tmpnorm;
            if(i != j && symm == 'S')
                norm += tmpnorm*tmpnorm;
        }
        info = Array_dot(M->far_U[bi], M->far_V[bi], &A2);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        info = Array_diff(A, A2, &tmpdiff);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        double tmperr = tmpdiff/tmpnorm;
        info = Array_free(A2);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        #pragma omp critical
        {
            diff += tmpdiff*tmpdiff;
            if(i != j && symm == 'S')
                diff += tmpdiff*tmpdiff;
            if(tmperr > maxerr)
                maxerr = tmperr;
        }
        info = Array_free(A);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
    }
    if(abort != 0)
        return abort;
    if(M->onfly == 0)
        #pragma omp parallel for
        for(bi = 0; bi < nblocks_near; bi++)
        {
            if(abort != 0)
                continue;
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            double tmpnorm;
            int info = Array_norm(M->near_D[bi], &tmpnorm);
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
            #pragma omp critical
            {
                norm += tmpnorm*tmpnorm;
                if(i != j && symm == 'S')
                    norm += tmpnorm*tmpnorm;
            }
        }
    else
        #pragma omp parallel for
        for(bi = 0; bi < nblocks_near; bi++)
        {
            if(abort != 0)
                continue;
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            int nrowsi = R->size[i];
            int ncolsj = C->size[j];
            int shape[2] = {nrowsi, ncolsj};
            Array *A;
            int info = Array_new(&A, 2, shape, P->dtype, 'F');
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
            info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                    C->pivot+C->start[j], P->row_data, P->col_data, A->data);
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
            double tmpnorm;
            info = Array_norm(A, &tmpnorm);
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
            #pragma omp critical
            {
                norm += tmpnorm*tmpnorm;
                if(i != j && symm == 'S')
                    norm += tmpnorm*tmpnorm;
            }
            info = Array_free(A);
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
        }
    if(abort != 0)
        return abort;
    tmp_time2 = omp_get_wtime();
    STARS_WARNING("total time: %f sec", tmp_time2-tmp_time);
    printf("Relative error of approximation of full matrix: %e\n",
            sqrt(diff/norm));
    printf("Maximum relative error of per-block approximation: %e\n", maxerr);
    return 0;
}

int STARS_BLRM_get_block(STARS_BLRM *M, int i, int j, int *shape, int *rank,
        void **U, void **V, void **D)
// Returns shape of block, its rank and low-rank factors or dense
// representation of a block
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(shape == NULL)
    {
        STARS_ERROR("invalid value of `shape`");
        return 1;
    }
    if(rank == NULL)
    {
        STARS_ERROR("invalid value of `rank`");
        return 1;
    }
    if(U == NULL)
    {
        STARS_ERROR("invalid value of `U`");
        return 1;
    }
    if(V == NULL)
    {
        STARS_ERROR("invalid value of `V`");
        return 1;
    }
    if(D == NULL)
    {
        STARS_ERROR("invalid value of `D`");
        return 1;
    }
    STARS_BLRF *F = M->blrf;
    if(i < 0 || i >= F->nbrows)
    {
        STARS_ERROR("invalid value of `i`");
        return 1;
    }
    if(j < 0 || j >= F->nbcols)
    {
        STARS_ERROR("invalid value of `j`");
        return 1;
    }
    STARS_Problem *P = F->problem;
    if(P->ndim != 2)
    {
        STARS_ERROR("only scalar kernels are supported");
        return 1;
    }
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    int nrows = R->size[i], ncols = C->size[j], info = 0;
    shape[0] = nrows;
    shape[1] = ncols;
    *rank = nrows < ncols ? nrows : ncols;
    *U = NULL;
    *V = NULL;
    *D = NULL;
    size_t bi = -1, k = F->brow_far_start[i];
    while(k < F->brow_far_start[i+1])
    {
        if(F->block_far[2*F->brow_far[k]+1] == j)
        {
            bi = k;
            break;
        }
        k++;
    }
    if(bi != -1)
    {
        *rank = M->far_rank[bi];
        *U = M->far_U[bi];
        *V = M->far_V[bi];
        return info;
    }
    k = F->brow_near_start[i];
    while(k < F->brow_near_start[i+1])
    {
        if(F->block_near[2*F->brow_near[k]+1] == j)
        {
            bi = k;
            break;
        }
        k++;
    }
    if(bi != -1)
    {
        *D = M->near_D[bi];
        if(*D == NULL)
            info = STARS_BLRF_get_block(F, i, j, shape, D);
        return info;
    }
    STARS_WARNING("Required block (%d, %d) is not admissible!\n", i, j);
    info = STARS_BLRF_get_block(F, i, j, shape, D);
    return info;
}

int STARS_BLRM_tiled_compress_algebraic_svd(STARS_BLRM **M, STARS_BLRF *F,
        int maxrank, double tol, int onfly)
// Private function of STARS-H
// Uses SVD to acquire rank of each block, compresses given matrix (given
// by block kernel, which returns submatrices) with relative accuracy tol
// or with given maximum rank (if maxrank == 0, then it is ignored)
{
    double tmp_time = omp_get_wtime(), tmp_time2;
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(F == NULL)
    {
        STARS_ERROR("invalid value of `F`");
        return 1;
    }
    if(maxrank < 0)
    {
        STARS_ERROR("invalid value of `maxrank`");
        return 1;
    }
    if(tol < 0 || tol >= 1)
    {
        STARS_ERROR("invalid value of `tol`");
        return 1;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARS_ERROR("invalid value of `onfly`");
        return 1;
    }
    STARS_Problem *P = F->problem;
    int ndim = P->ndim;
    if(ndim != 2)
    {
        STARS_ERROR("only scalar kernels are supported");
        return 1;
    }
    STARS_BLRF *F2;
    Array *A, *A2;
    Array *U, *S, *V;
    double *ptr, *ptrS, *ptrV;
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near;
    // Following values are used when there are false far-field blocks. Default
    // value are old sizes.
    size_t new_nblocks_far = nblocks_far, new_nblocks_near = nblocks_near;
    int *block_far = F->block_far, *block_near = F->block_near;
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank = NULL;
    if(nblocks_far > 0)
    {
        STARS_MALLOC(far_U, nblocks_far);
        STARS_MALLOC(far_V, nblocks_far);
        STARS_MALLOC(far_rank, nblocks_far);
    }
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    int nrowsi, ncolsj;
    int shape[ndim];
    size_t bi, bj, k, l;
    int i, j, mn, rank, info;
    for(bi = 0; bi < nblocks_far; bi++)
    // Cycle over every admissible block, `bj` stores current number of false
    // far-bield blocks
    {
        i = block_far[2*bi];
        j = block_far[2*bi+1];
        nrowsi = R->size[i];
        ncolsj = C->size[j];
        shape[0] = nrowsi;
        shape[ndim-1] = ncolsj;
        mn = nrowsi > ncolsj ? ncolsj : nrowsi;
        info = Array_new(&A, ndim, shape, P->dtype, 'F');
        if(info != 0)
            return info;
        info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                C->pivot+C->start[j], P->row_data, P->col_data, A->data);
        if(info != 0)
            return info;
        info = Array_new_copy(&A2, A, 'N');
        if(info != 0)
            return info;
        info = Array_SVD(A, &U, &S, &V);
        if(info != 0)
            return info;
        info = SVD_get_rank(S, tol, 'F', &rank);
        if(info != 0)
            return info;
        info = Array_free(A);
        if(info != 0)
            return info;
        if(rank < mn/2)
        // If block is low-rank
        {
            if(rank > maxrank)
                rank = maxrank;
            far_rank[bi] = rank;
            shape[0] = nrowsi;
            shape[1] = rank;
            info = Array_new(far_U+bi, 2, shape, 'd', 'F');
            if(info != 0)
                return info;
            shape[0] = rank;
            shape[1] = ncolsj;
            info = Array_new(far_V+bi, 2, shape, 'd', 'F');
            if(info != 0)
                return info;
            cblas_dcopy(rank*nrowsi, U->data, 1, far_U[bi]->data, 1);
            ptr = far_V[bi]->data;
            ptrS = S->data;
            ptrV = V->data;
            for(k = 0; k < ncolsj; k++)
                for(l = 0; l < rank; l++)
                {
                    ptr[k*rank+l] = ptrS[l]*ptrV[k*mn+l];
                }
            info = Array_free(A2);
            if(info != 0)
                return info;
        }
        else
        // If block is NOT low-rank
        {
            if(onfly == 0)
                far_U[bi] = A2;
            else
            {
                info = Array_free(A2);
                if(info != 0)
                    return info;
            }
            far_rank[bi] = -1;
        }
        info = Array_free(U);
        if(info != 0)
            return info;
        info = Array_free(S);
        if(info != 0)
            return info;
        info = Array_free(V);
        if(info != 0)
            return info;
    }
    size_t nblocks_false_far = 0;
    size_t *false_far = NULL;
    for(bi = 0; bi < nblocks_far; bi++)
        if(far_rank[bi] == -1)
            nblocks_false_far++;
    STARS_MALLOC(false_far, nblocks_false_far);
    bj = 0;
    for(bi = 0; bi < nblocks_far; bi++)
        if(far_rank[bi] == -1)
            false_far[bj++] = bi;
    // Update lists of far-field and near-field blocks and replace `blrf` field
    // if required (if update was actually done). After finishing this scope,
    // far-field blocks are done. Need to generate near-field blocks.
    if(nblocks_false_far > 0)
    {
        new_nblocks_near = nblocks_near+nblocks_false_far;
        STARS_MALLOC(block_near, 2*new_nblocks_near);
        for(bi = 0; bi < 2*nblocks_near; bi++)
            block_near[bi] = F->block_near[bi];
        for(bi = 0; bi < nblocks_false_far; bi++)
        {
            bj = false_far[bi];
            block_near[2*(bi+nblocks_near)] = F->block_far[2*bj];
            block_near[2*(bi+nblocks_near)+1] = F->block_far[2*bj+1];
        }
        //nblocks_near = new_nblocks_near;
        new_nblocks_far = nblocks_far-nblocks_false_far;
        if(new_nblocks_far > 0)
        {
            STARS_MALLOC(block_far, 2*new_nblocks_far);
            bj = 0;
            for(bi = 0; bi < nblocks_far; bi++)
            {
                if(false_far[bj] == bi)
                    bj++;
                else
                {
                    block_far[2*(bi-bj)] = F->block_far[2*bi];
                    block_far[2*(bi-bj)+1] = F->block_far[2*bi+1];
                }
            }
        }
        else
        {
            block_far = NULL;
            free(far_rank);
            far_rank = NULL;
            free(far_U);
            far_U = NULL;
            free(far_V);
            far_V = NULL;
        }
        //nblocks_far = new_nblocks_far;
        info = STARS_BLRF_new(&F2, P, F->symm, R, C, new_nblocks_far,
                block_far, new_nblocks_near, block_near, F->type);
        if(info != 0)
            return info;
        STARS_BLRF_swap(F, F2);
        STARS_WARNING("`F` was modified due to false far-field blocks");
        info = STARS_BLRF_free(F2);
        if(info != 0)
            return info;
    }
    // Compute near-filed blocks if needed and false far-field blocks to them
    if(onfly == 0 && new_nblocks_near > 0)
    {

        STARS_MALLOC(near_D, new_nblocks_near);
        // At first work with old near-field blocks
        // And then with false far-field blocks
        for(bi = 0; bi < nblocks_near; bi++)
        {
            i = block_near[2*bi];
            j = block_near[2*bi+1];
            nrowsi = R->size[i];
            ncolsj = C->size[j];
            shape[0] = nrowsi;
            shape[ndim-1] = ncolsj;
            mn = nrowsi > ncolsj ? ncolsj : nrowsi;
            Array *A;
            info = Array_new(&A, ndim, shape, P->dtype, 'F');
            info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                    C->pivot+C->start[j], P->row_data, P->col_data,
                    A->data);
            near_D[bi] = A;
        }
        for(bi = nblocks_near; bi < new_nblocks_near; bi++)
        {
            bj = false_far[bi-nblocks_near];
            near_D[bi] = far_U[bj];
        }
    }
    // Changing size of far_rank, far_U and far_V
    if(nblocks_false_far > 0)
    {
        bj = 0;
        for(bi = 0; bi < nblocks_far; bi++)
        {
            if(false_far[bj] == bi)
                bj++;
            else
            {
                far_U[bi-bj] = far_U[bi];
                far_V[bi-bj] = far_V[bi];
                far_rank[bi-bj] = far_rank[bi];
            }
        }
        STARS_REALLOC(far_rank, new_nblocks_far);
        STARS_REALLOC(far_U, new_nblocks_far);
        STARS_REALLOC(far_V, new_nblocks_far);
    }
    free(false_far);
    tmp_time2 = omp_get_wtime();
    STARS_WARNING("total time: %f sec", tmp_time2-tmp_time);
    return STARS_BLRM_new(M, F, far_rank, far_U, far_V, onfly, near_D,
            NULL, NULL, NULL, '2');
}

int STARS_BLRM_tiled_compress_algebraic_svd_ompfor(STARS_BLRM **M,
        STARS_BLRF *F, int maxrank, double tol, int onfly)
// Private function of STARS-H
// Uses SVD to acquire rank of each block, compresses given matrix (given
// by block kernel, which returns submatrices) with relative accuracy tol
// or with given maximum rank (if maxrank == 0, then it is ignored)
{
    double tmp_time = omp_get_wtime(), tmp_time2;
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(F == NULL)
    {
        STARS_ERROR("invalid value of `F`");
        return 1;
    }
    if(maxrank < 0)
    {
        STARS_ERROR("invalid value of `maxrank`");
        return 1;
    }
    if(tol < 0 || tol >= 1)
    {
        STARS_ERROR("invalid value of `tol`");
        return 1;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARS_ERROR("invalid value of `onfly`");
        return 1;
    }
    STARS_Problem *P = F->problem;
    int ndim = P->ndim;
    if(ndim != 2)
    {
        STARS_ERROR("only scalar kernels are supported");
        return 1;
    }
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near;
    // Following values are used when there are false far-field blocks. Default
    // value are old sizes.
    size_t new_nblocks_far = nblocks_far, new_nblocks_near = nblocks_near;
    int *block_far = F->block_far, *block_near = F->block_near;
    size_t nblocks_false_far = 0;
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    size_t *false_far = NULL;
    int *far_rank = NULL;
    if(nblocks_far > 0)
    {
        STARS_MALLOC(far_U, nblocks_far);
        STARS_MALLOC(far_V, nblocks_far);
        STARS_MALLOC(far_rank, nblocks_far);
    }
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    int info;
    int shape[ndim];
    size_t bi, bj = 0;
    volatile int abort = 0;
    #pragma omp parallel for
    for(bi = 0; bi < nblocks_far; bi++)
    // Cycle over every admissible block, `bj` stores current number of false
    // far-bield blocks
    {
        #pragma omp flush(abort)
        if(abort != 0)
            continue;
        int i = block_far[2*bi];
        int j = block_far[2*bi+1];
        int nrowsi = R->size[i];
        int ncolsj = C->size[j];
        int shape[ndim];
        shape[0] = nrowsi;
        shape[ndim-1] = ncolsj;
        int mn = nrowsi > ncolsj ? ncolsj : nrowsi;
        int rank;
        Array *A, *A2, *U, *S, *V;
        int info = Array_new(&A, ndim, shape, P->dtype, 'F');
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                C->pivot+C->start[j], P->row_data, P->col_data, A->data);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        info = Array_new_copy(&A2, A, 'N');
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        info = Array_SVD(A, &U, &S, &V);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        info = SVD_get_rank(S, tol, 'F', &rank);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        info = Array_free(A);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        if(rank < mn/2)
        // If block is low-rank
        {
            if(rank > maxrank)
                rank = maxrank;
            shape[0] = nrowsi;
            shape[1] = rank;
            info = Array_new(far_U+bi, 2, shape, 'd', 'F');
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
            shape[0] = rank;
            shape[1] = ncolsj;
            info = Array_new(far_V+bi, 2, shape, 'd', 'F');
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
            cblas_dcopy(rank*nrowsi, U->data, 1, far_U[bi]->data, 1);
            double *ptr = far_V[bi]->data;
            double *ptrS = S->data;
            double *ptrV = V->data;
            for(size_t k = 0; k < ncolsj; k++)
                for(size_t l = 0; l < rank; l++)
                {
                    ptr[k*rank+l] = ptrS[l]*ptrV[k*mn+l];
                }
            far_rank[bi] = rank;
            info = Array_free(A2);
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
        }
        else
        // If block is NOT low-rank
        {
            if(abort != 0)
                continue;
            far_rank[bi] = -1;
            if(onfly == 0)
                far_U[bi] = A2;
            else
            {
                info = Array_free(A2);
                if(info != 0)
                {
                    #pragma omp atomic write
                    abort = info;
                    continue;
                }
            }
        }
        info = Array_free(U);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        info = Array_free(S);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
        info = Array_free(V);
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
    }
    if(abort != 0)
        return abort;
    for(bi = 0; bi < nblocks_far; bi++)
        if(far_rank[bi] == -1)
            nblocks_false_far++;
    STARS_MALLOC(false_far, nblocks_false_far);
    bj = 0;
    for(bi = 0; bi < nblocks_far; bi++)
        if(far_rank[bi] == -1)
            false_far[bj++] = bi;
    // Update lists of far-field and near-field blocks and replace `blrf` field
    // if required (if update was actually done). After finishing this scope,
    // far-field blocks are done. Need to generate near-field blocks.
    if(nblocks_false_far > 0)
    {
        new_nblocks_near = nblocks_near+nblocks_false_far;
        STARS_MALLOC(block_near, 2*new_nblocks_near);
        #pragma omp parallel for
        for(bi = 0; bi < 2*nblocks_near; bi++)
            block_near[bi] = F->block_near[bi];
        #pragma omp parallel for
        for(bi = 0; bi < nblocks_false_far; bi++)
        {
            size_t bj = false_far[bi];
            block_near[2*(bi+nblocks_near)] = F->block_far[2*bj];
            block_near[2*(bi+nblocks_near)+1] = F->block_far[2*bj+1];
        }
        new_nblocks_far = nblocks_far-nblocks_false_far;
        if(new_nblocks_far > 0)
        {
            STARS_MALLOC(block_far, 2*new_nblocks_far);
            bj = 0;
            for(bi = 0; bi < nblocks_far; bi++)
            {
                if(false_far[bj] == bi)
                {
                    bj++;
                }
                else
                {
                    block_far[2*(bi-bj)] = F->block_far[2*bi];
                    block_far[2*(bi-bj)+1] = F->block_far[2*bi+1];
                }
            }
        }
        else
        {
            block_far = NULL;
            free(far_rank);
            far_rank = NULL;
            free(far_U);
            far_U = NULL;
            free(far_V);
            far_V = NULL;
        }
        STARS_BLRF *F2;
        info = STARS_BLRF_new(&F2, P, F->symm, R, C, new_nblocks_far,
                block_far, new_nblocks_near, block_near, F->type);
        STARS_BLRF_swap(F, F2);
        STARS_WARNING("`F` was modified due to false far-field blocks");
        info = STARS_BLRF_free(F2);
        if(info != 0)
            return info;
    }
    // Compute near-filed blocks if needed and false far-field blocks to them
    if(onfly == 0 && new_nblocks_near > 0)
    {
        STARS_MALLOC(near_D, new_nblocks_near);
        // At first work with old near-field blocks
        // And then with false far-field blockso
        #pragma omp parallel for
        for(bi = 0; bi < nblocks_near; bi++)
        {
            if(abort != 0)
                continue;
            int i = block_near[2*bi];
            int j = block_near[2*bi+1];
            int nrowsi = R->size[i];
            int ncolsj = C->size[j];
            int shape[ndim];
            shape[0] = nrowsi;
            shape[ndim-1] = ncolsj;
            int mn = nrowsi > ncolsj ? ncolsj : nrowsi;
            Array *A;
            int info = Array_new(&A, ndim, shape, P->dtype, 'F');
            info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                    C->pivot+C->start[j], P->row_data, P->col_data,
                    A->data);
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
            near_D[bi] = A;
        }
        if(abort != 0)
            return abort;
        #pragma omp parallel for
        for(bi = nblocks_near; bi < new_nblocks_near; bi++)
        {
            size_t bj = false_far[bi-nblocks_near];
            near_D[bi] = far_U[bj];
        }
    }
    // Changing size of far_rank, far_U and far_V
    if(nblocks_false_far > 0)
    {
        bj = 0;
        for(bi = 0; bi < nblocks_far; bi++)
        {
            if(false_far[bj] == bi)
                bj++;
            else
            {
                far_U[bi-bj] = far_U[bi];
                far_V[bi-bj] = far_V[bi];
                far_rank[bi-bj] = far_rank[bi];
            }
        }
        STARS_REALLOC(far_rank, new_nblocks_far);
        STARS_REALLOC(far_U, new_nblocks_far);
        STARS_REALLOC(far_V, new_nblocks_far);
    }
    free(false_far);
    tmp_time2 = omp_get_wtime();
    STARS_WARNING("total time: %f sec", tmp_time2-tmp_time);
    return STARS_BLRM_new(M, F, far_rank, far_U, far_V, onfly, near_D,
            NULL, NULL, NULL, '2');
}

int STARS_BLRM_tiled_compress_algebraic_svd_ompfor2(STARS_BLRM **M,
        STARS_BLRF *F, int maxrank, double tol, int onfly)
// Private function of STARS-H
// Uses SVD to acquire rank of each block, compresses given matrix (given
// by block kernel, which returns submatrices) with relative accuracy tol
// or with given maximum rank (if maxrank <= 0, then tolerance is used)
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(F == NULL)
    {
        STARS_ERROR("invalid value of `F`");
        return 1;
    }
    if(maxrank < 0)
    {
        STARS_ERROR("invalid value of `maxrank`");
        return 1;
    }
    if(tol < 0 && tol >= 1)
    {
        STARS_ERROR("invalid value of `tol`");
        return 1;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARS_ERROR("invalid value of `onfly`");
        return 1;
    }
    double total_time = omp_get_wtime();
    STARS_Problem *P = F->problem;
    int ndim = P->ndim;
    if(ndim != 2)
    {
        STARS_ERROR("only scalar kernels are supported");
        return 1;
    }
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near;
    Array **far_U, **far_V;
    STARS_MALLOC(far_U, nblocks_far);
    STARS_MALLOC(far_V, nblocks_far);
    int *far_rank;
    STARS_MALLOC(far_rank, nblocks_far);
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    //#pragma omp parallel
    {
        size_t bi, k, l;
        Array *A, *A2, *U, *S, *V;
        double *ptr, *ptrS, *ptrV;
        int nrowsi, ncolsj, i, j, ndim = P->ndim, mn, rank, info;
        int *shape;
        STARS_MALLOC(shape, ndim);
        memcpy(shape, P->shape, ndim*sizeof(*shape));
        //#pragma omp for
        for(bi = 0; bi < nblocks_far; bi++)
        // Cycle over every admissible block
        {
            i = F->block_far[2*bi];
            j = F->block_far[2*bi+1];
            nrowsi = R->size[i];
            ncolsj = C->size[j];
            shape[0] = nrowsi;
            shape[ndim-1] = ncolsj;
            mn = nrowsi > ncolsj ? ncolsj : nrowsi;
            far_U[bi] = NULL;
            far_V[bi] = NULL;
            info = Array_new(&A, ndim, shape, P->dtype, 'F');
            if(info != 0)
                return info;
            info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                    C->pivot+C->start[j], P->row_data, P->col_data, A->data);
            if(info != 0)
                return info;
            info = Array_new_copy(&A2, A, 'N');
            if(info != 0)
                return info;
            info = Array_SVD(A, &U, &S, &V);
            if(info != 0)
                return info;
            info = SVD_get_rank(S, tol, 'F', &rank);
            if(info != 0)
                return info;
            info = Array_free(A);
            if(info != 0)
                return info;
            if(rank < mn/2)
            // If block is low-rank
            {
                if(rank > maxrank)
                    rank = maxrank;
                shape[0] = nrowsi;
                shape[1] = rank;
                info = Array_new(far_U+bi, 2, shape, 'd', 'F');
                if(info != 0)
                    return info;
                shape[0] = rank;
                shape[1] = ncolsj;
                info = Array_new(far_V+bi, 2, shape, 'd', 'F');
                if(info != 0)
                    return info;
                cblas_dcopy(rank*nrowsi, U->data, 1, far_U[bi]->data, 1);
                ptr = far_V[bi]->data;
                ptrS = S->data;
                ptrV = V->data;
                for(k = 0; k < ncolsj; k++)
                    for(l = 0; l < rank; l++)
                    {
                        ptr[k*rank+l] = ptrS[l]*ptrV[k*mn+l];
                    }
                far_rank[bi] = rank;
                info = Array_free(A2);
                if(info != 0)
                    return info;
            }
            else
            // If block is NOT low-rank
            {
                info = Array_free(A2);
                if(info != 0)
                    return info;
                far_rank[bi] = mn;
            }
            info = Array_free(U);
            if(info != 0)
                return info;
            info = Array_free(S);
            if(info != 0)
                return info;
            info = Array_free(V);
            if(info != 0)
                return info;
        }
        free(shape);
    }
    /*
    if(onfly == 0)
    #pragma omp parallel private(bi, i, j, nrowsi, ncolsj, mn, block)
    {
        int *shape = malloc(ndim*sizeof(int));
        memcpy(shape, B->P->shape, ndim*sizeof(int));
        #pragma omp for
        for(bi = 0; bi < nblocks_near; bi++)
        {
            i = B->block_near[2*bi];
            j = B->block_near[2*bi+1];
            nrowsi = R->size[i];
            ncolsj = C->size[j];
            shape[0] = nrowsi;
            shape[ndim-1] = ncolsj;
            mn = nrowsi > ncolsj ? ncolsj : nrowsi;
            block = Array_new(ndim, shape, P->dtype, 'F');
            (P->kernel)(nrowsi, ncolsj, R->pivot+
                    R->start[i], C->pivot+
                    C->start[j], P->row_data,
                    P->col_data, block->buffer);
            near_D[bi] = block;
        }
        free(shape);
    }
    */
    printf("TOTAL TIME: %f\n", omp_get_wtime()-total_time);
    return STARS_BLRM_new(M, F, far_rank, far_U, far_V, 1, NULL,
            NULL, NULL, NULL, '2');
}

int STARS_BLRM_tiled_compress_algebraic_svd_batched(STARS_BLRM **M,
        STARS_BLRF *F, int maxrank, double tol, int onfly,
        size_t max_buffer_size)
{
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(F == NULL)
    {
        STARS_ERROR("invalid value of `F`");
        return 1;
    }
    if(maxrank < 0)
    {
        STARS_ERROR("invalid value of `maxrank`");
        return 1;
    }
    if(tol < 0 && tol >= 1)
    {
        STARS_ERROR("invalid value of `tol`");
        return 1;
    }
    if(onfly != 0 && onfly != 1)
    {
        STARS_ERROR("invalid value of `onfly`");
        return 1;
    }
    if(max_buffer_size == 0)
    {
        STARS_ERROR("invalid value of `max_buffer_size`");
        return 1;
    }
    double total_time = omp_get_wtime();
    STARS_Problem *P = F->problem;
    size_t bbi, bi;
    int ndim = P->ndim;
    if(ndim != 2)
    {
        STARS_ERROR("only scalar kernels are supported");
        return 1;
    }
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near;
    Array **far_U, **far_V;
    STARS_MALLOC(far_U, nblocks_far);
    STARS_MALLOC(far_V, nblocks_far);
    int *shape = P->shape, *far_rank;
    void *alloc_U, *alloc_V;
    size_t alloc_U_size = 2*(size_t)F->nbcols*(size_t)shape[0]*
            (size_t)maxrank*sizeof(double);
    size_t alloc_V_size = 2*(size_t)F->nbrows*(size_t)shape[1]*
            (size_t)maxrank*sizeof(double);
    STARS_MALLOC(alloc_U, alloc_U_size);
    STARS_MALLOC(alloc_V, alloc_V_size);
    void *current_U = alloc_U, *current_V = alloc_V;
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    STARS_MALLOC(far_rank, nblocks_far);
    void *row_data = P->row_data, *col_data = P->col_data;
    size_t *ltotalwork, *lwork_arrays;
    STARS_MALLOC(ltotalwork, nblocks_far);
    STARS_MALLOC(lwork_arrays, 5*nblocks_far);
    size_t *lbwork = lwork_arrays;
    size_t *luvwork = lwork_arrays+nblocks_far;
    size_t *lwork = lwork_arrays+2*nblocks_far;
    size_t *lswork = lwork_arrays+3*nblocks_far;
    size_t *liwork = lwork_arrays+4*nblocks_far;
    void *tmp_buffer;
    STARS_MALLOC(tmp_buffer, max_buffer_size);
    double tmp_time = omp_get_wtime(), tmp_time2;
    //#pragma omp parallel for
    for(bi = 0; bi < nblocks_far; bi++)
    {
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        size_t nrowsi = R->size[i], ncolsj = C->size[j];
        if(nrowsi < ncolsj)
        {
            lbwork[bi] = nrowsi*ncolsj;
            luvwork[bi] = nrowsi*nrowsi;
            i = (5*nrowsi+7)*nrowsi;
            j = 3*nrowsi+ncolsj;
            lwork[bi] = i;
            if(i < j)
                lwork[bi] = j;
            liwork[bi] = 8*nrowsi;
            lswork[bi] = nrowsi;
            ltotalwork[bi] = (lbwork[bi]+luvwork[bi]+lwork[bi]+lswork[bi])*
                    sizeof(double)+liwork[bi]*sizeof(int);
        }
        else
        {
            lbwork[bi] = nrowsi*ncolsj;
            luvwork[bi] = ncolsj*ncolsj;
            i = (5*ncolsj+7)*ncolsj;
            j = 3*ncolsj+nrowsi;
            lwork[bi] = i;
            if(i < j)
                lwork[bi] = j;
            liwork[bi] = 8*ncolsj;
            lswork[bi] = ncolsj;
            ltotalwork[bi] = (lbwork[bi]+luvwork[bi]+lwork[bi]+lswork[bi])*
                    sizeof(double)+liwork[bi]*sizeof(int);
        }
    }
    //tmp_time2 = omp_get_wtime();
    //printf("TIME1: %f\n", tmp_time2-tmp_time);
    //tmp_time = tmp_time2;
    size_t nblocks_processed = 0;
    while(nblocks_processed < nblocks_far)
    {
        //printf("%d %d\n", nblocks_processed, nblocks_far);
        size_t tmp_ltotalwork = 0;
        size_t tmp_lbwork = 0, tmp_luvwork = 0, tmp_lwork = 0, tmp_lswork = 0;
        bi = nblocks_processed;
        while(bi < nblocks_far && tmp_ltotalwork+ltotalwork[bi] <
                max_buffer_size)
        {
            tmp_ltotalwork += ltotalwork[bi];
            tmp_lbwork += lbwork[bi];
            tmp_luvwork += luvwork[bi];
            tmp_lwork += lwork[bi];
            tmp_lswork += lswork[bi];
            bi++;
        }
        size_t batch_size = bi-nblocks_processed;
        int nrows[batch_size], ncols[batch_size];
        int *irow[batch_size], *icol[batch_size];
        double *buffer[batch_size];
        double *U[batch_size], *S[batch_size], *V[batch_size], *UV[batch_size];
        // following is int and int * for compatibility with LAPACKE
        int ldv[batch_size], *iwork[batch_size];
        double *work[batch_size];
        //printf("batch size %d, size %u maxsize %u\n", batch_size,
        //        tmp_buffer_size, max_buffer_size);
        buffer[0] = tmp_buffer;
        UV[0] = buffer[0]+tmp_lbwork;
        S[0] = UV[0]+tmp_luvwork;
        work[0] = S[0]+tmp_lswork;
        iwork[0] = (int *)(work[0]+tmp_lwork);
        for(bbi = 0; bbi < batch_size-1; bbi++)
        {
            buffer[bbi+1] = buffer[bbi]+lbwork[bbi+nblocks_processed];
            UV[bbi+1] = UV[bbi]+luvwork[bbi+nblocks_processed];
            S[bbi+1] = S[bbi]+lswork[bbi+nblocks_processed];
            work[bbi+1] = work[bbi]+lwork[bbi+nblocks_processed];
            iwork[bbi+1] = iwork[bbi]+liwork[bbi+nblocks_processed];
        }
        //tmp_time2 = omp_get_wtime();
        //printf("TIME2: %f\n", tmp_time2-tmp_time);
        //tmp_time = tmp_time2;
        //#pragma omp parallel for
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            size_t bi = bbi+nblocks_processed;
            int i = F->block_far[2*bi];
            int j = F->block_far[2*bi+1];
            nrows[bbi] = R->size[i];
            ncols[bbi] = C->size[j];
            irow[bbi] = R->pivot+R->start[i];
            icol[bbi] = C->pivot+C->start[j];
            if(nrows[bbi] < ncols[bbi])
            {
                U[bbi] = UV[bbi];
                V[bbi] = buffer[bbi];
                ldv[bbi] = nrows[bbi];
            }
            else
            {
                U[bbi] = buffer[bbi];
                V[bbi] = UV[bbi];
                ldv[bbi] = ncols[bbi];
            }
        }
        //tmp_time2 = omp_get_wtime();
        //printf("TIME3: %f\n", tmp_time2-tmp_time);
        //tmp_time = tmp_time2;
        //printf("batch size %d\n", batch_size);
        //#pragma omp parallel for
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            int info = P->kernel(nrows[bbi], ncols[bbi], irow[bbi], icol[bbi],
                    row_data, col_data, buffer[bbi]);
            if(info != 0)
                return info;
        }
        //tmp_time2 = omp_get_wtime();
        //printf("TIME4: %f\n", tmp_time2-tmp_time);
        //tmp_time = tmp_time2;
        //printf("DONE WITH KERNEL\n");
        //#pragma omp parallel for
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            //LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, jobu[bbi], jobv[bbi],
            //        nrows[bbi], ncols[bbi], buffer[bbi], nrows[bbi], S[bbi],
            //        U[bbi], nrows[bbi], V[bbi], ldv[bbi], work[bbi],
            //        lwork[bbi]);
            int info = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'O', nrows[bbi],
                    ncols[bbi], buffer[bbi], nrows[bbi], S[bbi], U[bbi],
                    nrows[bbi], V[bbi], ldv[bbi], work[bbi],
                    lwork[bbi+nblocks_processed], iwork[bbi]);
            if(info != 0)
                return info;
        }
        //printf("DONE WITH SVD\n");
        //tmp_time2 = omp_get_wtime();
        //printf("TIME5: %f\n", tmp_time2-tmp_time);
        //tmp_time = tmp_time2;
        //#pragma omp parallel for
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            size_t bi = bbi+nblocks_processed;
            double *ptrS = S[bbi];
            double Stol = 0, Stmp = 0.;
            size_t i, j, mn = ldv[bbi], rank = mn;
            int shapeU[2] = {nrows[bbi], 0}, shapeV[2] = {0, ncols[bbi]}, info;
            for(i = 0; i < mn; i++)
                Stol += ptrS[i]*ptrS[i];
            Stol *= tol*tol;
            while(rank > 1 && Stol > Stmp)
            {
                rank--;
                Stmp += ptrS[rank]*ptrS[rank];
            }
            //printf("rank %d\n", rank);
            rank++;
            if(2*rank >= mn)
            {
                far_rank[bi] = mn;
                far_U[bi] = NULL;
                far_V[bi] = NULL;
            }
            else
            {
                if(rank > maxrank)
                    rank = maxrank;
                far_rank[bi] = rank;
                shapeU[1] = rank;
                shapeV[0] = rank;
                //#pragma omp critical
                {
                    info = Array_from_buffer(far_U+bi, 2, shapeU, 'd', 'F',
                            current_U);
                    if(info != 0)
                        return info;
                    current_U += 2*(size_t)shapeU[0]*(size_t)shapeU[1]*
                            sizeof(double);
                    info = Array_from_buffer(far_V+bi, 2, shapeV, 'd', 'F',
                            current_V);
                    if(info != 0)
                        return info;
                    current_V += 2*(size_t)shapeV[0]*(size_t)shapeV[1]*
                            sizeof(double);
                }
                // THIS MAY LEAD TO PROBLEM !! BETTER TO DO CYCLE OF COPIES
                cblas_dcopy((size_t)shapeU[0]*(size_t)shapeU[1], U[bbi], 1,
                        far_U[bi]->data, 1);
                double *ptr = far_V[bi]->data, *ptrV = V[bbi];
                for(i = 0; i < shapeV[1]; i++)
                    for(j = 0; j < shapeV[0]; j++)
                        ptr[i*rank+j] = ptrS[j]*ptrV[i*mn+j];
            }
        }
        //tmp_time2 = omp_get_wtime();
        //printf("TIME6: %f\n", tmp_time2-tmp_time);
        //tmp_time = tmp_time2;
        //printf("DONE WITH far_U far_V\n");
        nblocks_processed += batch_size;
    }
    free(tmp_buffer);
    free(ltotalwork);
    free(lwork_arrays);
    //if(onfly == 0)
    //    near_D = malloc(nblocks_near*sizeof(Array *));
    printf("TOTAL TIME: %f\n", omp_get_wtime()-total_time);
    return STARS_BLRM_new(M, F, far_rank, far_U, far_V, 1, NULL,
            alloc_U, alloc_V, NULL, '1');
}
