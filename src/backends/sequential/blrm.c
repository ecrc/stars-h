#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>
#include "stars.h"
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
//#include "cblas.h"
//#include "lapacke.h"
#include "misc.h"

int STARS_BLRM_error(STARS_BLRM *M)
// Measure error of approximation by non-nested block low-rank matrix
{
    struct timeval tmp_time, tmp_time2;
    gettimeofday(&tmp_time, NULL);
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
    // Do not use inner parallelism not to increase number of threads
    for(bi = 0; bi < nblocks_far; bi++)
    {
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        int nrowsi = R->size[i];
        int ncolsj = C->size[j];
        int shape[2] = {nrowsi, ncolsj};
        Array *A, *A2;
        int info = Array_new(&A, 2, shape, P->dtype, 'F');
        if(info != 0)
            return info;
        info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                C->pivot+C->start[j], P->row_data, P->col_data, A->data);
        if(info != 0)
            return info;
        double tmpnorm, tmpdiff;
        info = Array_norm(A, &tmpnorm);
        if(info != 0)
            return info;
        norm += tmpnorm*tmpnorm;
        if(i != j && symm == 'S')
            norm += tmpnorm*tmpnorm;
        info = Array_dot(M->far_U[bi], M->far_V[bi], &A2);
        if(info != 0)
            return info;
        info = Array_diff(A, A2, &tmpdiff);
        if(info != 0)
            return info;
        double tmperr = tmpdiff/tmpnorm;
        info = Array_free(A2);
        if(info != 0)
            return info;
        diff += tmpdiff*tmpdiff;
        if(i != j && symm == 'S')
            diff += tmpdiff*tmpdiff;
        if(tmperr > maxerr)
            maxerr = tmperr;
        info = Array_free(A);
        if(info != 0)
            return info;
    }
    if(M->onfly == 0)
        for(bi = 0; bi < nblocks_near; bi++)
        {
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            double tmpnorm;
            int info = Array_norm(M->near_D[bi], &tmpnorm);
            if(info != 0)
                return info;
            norm += tmpnorm*tmpnorm;
            if(i != j && symm == 'S')
                norm += tmpnorm*tmpnorm;
        }
    else
        for(bi = 0; bi < nblocks_near; bi++)
        {
            int i = F->block_near[2*bi];
            int j = F->block_near[2*bi+1];
            int nrowsi = R->size[i];
            int ncolsj = C->size[j];
            int shape[2] = {nrowsi, ncolsj};
            Array *A;
            int info = Array_new(&A, 2, shape, P->dtype, 'F');
            if(info != 0)
                return info;
            info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                    C->pivot+C->start[j], P->row_data, P->col_data, A->data);
            if(info != 0)
                return info;
            double tmpnorm;
            info = Array_norm(A, &tmpnorm);
            if(info != 0)
                return info;
            norm += tmpnorm*tmpnorm;
            if(i != j && symm == 'S')
                norm += tmpnorm*tmpnorm;
            info = Array_free(A);
            if(info != 0)
                return info;
        }
    gettimeofday(&tmp_time2, NULL);
    double time = tmp_time2.tv_sec-tmp_time.tv_sec+
            (tmp_time2.tv_usec-tmp_time.tv_usec)*1e-6;
    STARS_WARNING("total time: %f sec", time);
    printf("Relative error of approximation of full matrix: %e\n",
            sqrt(diff/norm));
    printf("Maximum relative error of per-block approximation: %e\n", maxerr);
    return 0;
}

int STARS_BLRM_tiled_compress_algebraic_svd(STARS_BLRM **M, STARS_BLRF *F,
        int fixrank, double tol, int onfly)
// Uses SVD to acquire rank of each block, compresses given matrix (given
// by block kernel, which returns submatrices) with relative accuracy tol
{
    //Set timer
    struct timeval tmp_time, tmp_time2;
    gettimeofday(&tmp_time, NULL);
    // Check parameters
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
    if(fixrank < 0)
    {
        STARS_ERROR("invalid value of `fixrank`");
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
    // Following values default to given block low-rank format F, but they are
    // changed when there are false far-field blocks.
    size_t new_nblocks_far = nblocks_far, new_nblocks_near = nblocks_near;
    int *block_far = F->block_far, *block_near = F->block_near;
    // Places to store low-rank factors, dense blocks and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank = NULL;
    // Allocate memory for far-field blocks
    if(nblocks_far > 0)
    {
        STARS_MALLOC(far_U, nblocks_far);
        STARS_MALLOC(far_V, nblocks_far);
        STARS_MALLOC(far_rank, nblocks_far);
    }
    // Shortcuts to information about tiles
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    // Work variables
    int nrowsi, ncolsj;
    int shape[ndim];
    size_t bi, bj, k, l;
    int i, j, mn, rank, info;
    for(bi = 0; bi < nblocks_far; bi++)
    // Cycle over every far-field block
    {
        i = block_far[2*bi];
        j = block_far[2*bi+1];
        nrowsi = R->size[i];
        ncolsj = C->size[j];
        shape[0] = nrowsi;
        shape[ndim-1] = ncolsj;
        mn = nrowsi > ncolsj ? ncolsj : nrowsi;
        // Get array, holding necessary data
        info = Array_new(&A, ndim, shape, P->dtype, 'F');
        if(info != 0)
            return info;
        info = P->kernel(nrowsi, ncolsj, R->pivot+R->start[i],
                C->pivot+C->start[j], P->row_data, P->col_data, A->data);
        if(info != 0)
            return info;
        // Copy array to use it for SVD
        info = Array_new_copy(&A2, A, 'N');
        if(info != 0)
            return info;
        info = Array_SVD(A, &U, &S, &V);
        if(info != 0)
            return info;
        // Get rank
        info = SVD_get_rank(S, tol, 'F', &rank);
        if(info != 0)
            return info;
        info = Array_free(A);
        if(info != 0)
            return info;
        if(rank < mn/2)
        // If block is low-rank
        {
            if(fixrank != 0)
                rank = fixrank;
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
            // Copy part of `U` into low-rank factor `far_U`
            cblas_dcopy(rank*nrowsi, U->data, 1, far_U[bi]->data, 1);
            ptr = far_V[bi]->data;
            ptrS = S->data;
            ptrV = V->data;
            // Copy part of `S`*`V` into low-rank factor `far_V`
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
        // Free temporary SVD buffers
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
    // Get false far-field blocks to change corresponding data in block
    // low-rank format `F`
    size_t nblocks_false_far = 0;
    size_t *false_far = NULL;
    for(bi = 0; bi < nblocks_far; bi++)
        if(far_rank[bi] == -1)
            nblocks_false_far++;
    if(nblocks_false_far > 0)
    {
        // IMPORTANT: `false_far` must to be in ascending order for later code
        // to work normally
        STARS_MALLOC(false_far, nblocks_false_far);
        bj = 0;
        for(bi = 0; bi < nblocks_far; bi++)
            if(far_rank[bi] == -1)
                false_far[bj++] = bi;
    }
    // Update lists of far-field and near-field blocks using previously
    // generated list of false far-field blocks
    if(nblocks_false_far > 0)
    {
        // Update list of near-field blocks
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
        // Update list of far-field blocks
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
        // Update parameter `F` by creating new format and swapping it with
        // given `F`
        info = STARS_BLRF_new(&F2, P, F->symm, R, C, new_nblocks_far,
                block_far, new_nblocks_near, block_near, F->type);
        if(info != 0)
            return info;
        // Swap of lists of far-field and near-field blocks of `F` and `F2`
        STARS_BLRF_swap(F, F2);
        STARS_WARNING("`F` was modified due to false far-field blocks");
        info = STARS_BLRF_free(F2);
        if(info != 0)
            return info;
    }
    // Compute near-field blocks if needed
    if(onfly == 0 && new_nblocks_near > 0)
    {
        STARS_MALLOC(near_D, new_nblocks_near);
        // At first work with old near-field blocks
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
        // Add false far-field blocks
        for(bi = nblocks_near; bi < new_nblocks_near; bi++)
        {
            bj = false_far[bi-nblocks_near];
            near_D[bi] = far_U[bj];
        }
    }
    // Update far_rank, far_U and far_V and change their sizes
    if(nblocks_false_far > 0 && new_nblocks_far > 0)
    {
        bj = 0;
        for(bi = 0; bi < nblocks_far; bi++)
        {
            // `false_far` must be in ascending order
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
    if(new_nblocks_far == 0 && nblocks_far > 0)
    // If all far-field blocks are false, then dealloc buffers
    {
        block_far = NULL;
        free(far_rank);
        far_rank = NULL;
        free(far_U);
        far_U = NULL;
        free(far_V);
        far_V = NULL;
    }
    // Free temporary list if false far-field blocks
    if(nblocks_false_far > 0)
        free(false_far);
    gettimeofday(&tmp_time2, NULL);
    double time = tmp_time2.tv_sec-tmp_time.tv_sec+
            (tmp_time2.tv_usec-tmp_time.tv_usec)*1e-6;
    STARS_WARNING("total time: %f sec", time);
    // Init Block Low-Rank Matrix with given buffers
    return STARS_BLRM_new(M, F, far_rank, far_U, far_V, onfly, near_D,
            NULL, NULL, NULL, '2');
}

int STARS_BLRM_to_matrix(STARS_BLRM *M, Array **A)
// Creates copy of Block Low-rank Matrix in dense format
{
    // Check parameters
    if(M == NULL)
    {
        STARS_ERROR("invalid value of `M`");
        return 1;
    }
    if(A == NULL)
    {
        STARS_ERROR("invalid value of `A`");
        return 1;
    }
    STARS_BLRF *F = M->blrf;
    STARS_Problem *P = F->problem;
    STARS_Cluster *RC = F->row_cluster, *CC = F->col_cluster;
    int onfly = M->onfly;
    int info = Array_new(A, P->ndim, P->shape, P->dtype, 'F');
    if(info != 0)
        return info;
    Array *A2 = *A;
    double *ptrA = A2->data;
    int lda = A2->shape[0];
    size_t bi;
    // At first restore far-field blocks
    for(bi = 0; bi < F->nblocks_far; bi++)
    {
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        Array *B;
        int info = Array_dot(M->far_U[bi], M->far_V[bi], &B);
        if(info != 0)
            return info;
        int shape[2], rank;
        void *U, *V, *D;
        info = STARS_BLRM_get_block(M, i, j, shape, &rank, &U, &V, &D);
        if(U != M->far_U[bi]->data || V != M->far_V[bi]->data || D != NULL)
        {
            STARS_ERROR("bad BLRM_get_block()");
            info = 2;
        }
        if(info != 0)
            return info;
        double *ptrB = B->data, *localA = ptrA;
        localA += CC->start[j]*lda+RC->start[i];
        int ldb = B->shape[0];
        for(size_t k = 0; k < B->shape[0]; k++)
            for(size_t l = 0; l < B->shape[1]; l++)
                localA[l*lda+k] = ptrB[l*ldb+k];
        if(F->symm == 'S')
        {
            localA = ptrA;
            localA += CC->start[j]+RC->start[i]*lda;
            for(size_t k = 0; k < B->shape[0]; k++)
                for(size_t l = 0; l < B->shape[1]; l++)
                    localA[l+k*lda] = ptrB[l*ldb+k];
        }
    }
    // Restore near-field blocks
    for(bi = 0; bi < F->nblocks_near; bi++)
    {
        int i = F->block_near[2*bi];
        int j = F->block_near[2*bi+1];
        Array *B = NULL;
        if(onfly == 0)
            B = M->near_D[bi];
        int shape[2], rank;
        void *U, *V, *D;
        int info = STARS_BLRM_get_block(M, i, j, shape, &rank, &U, &V, &D);
        if(U != NULL || V != NULL || (onfly == 0 && D != B->data))
        {
            STARS_ERROR("bad BLRM_get_block()");
            info = 2;
        }
        if(info != 0)
            return info;
        double *ptrB = D, *localA = ptrA;
        localA += CC->start[j]*lda+RC->start[i];
        int ldb = shape[0];
        for(size_t k = 0; k < shape[0]; k++)
            for(size_t l = 0; l < shape[1]; l++)
                localA[l*lda+k] = ptrB[l*ldb+k];
        if(F->symm == 'S')
        {
            localA = ptrA;
            localA += CC->start[j]+RC->start[i]*lda;
            for(size_t k = 0; k < shape[0]; k++)
                for(size_t l = 0; l < shape[1]; l++)
                    localA[l+k*lda] = ptrB[l*ldb+k];
        }
        if(onfly == 1)
            free(D);
    }
    return 0;
}

int STARS_BLRM_heatmap(STARS_BLRM *M, char *filename)
{
    STARS_BLRF *F = M->blrf;
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    int *rank_map = malloc((size_t)F->nbrows*(size_t)F->nbcols*
            sizeof(*rank_map));
    size_t bi;
    for(bi = 0; bi < F->nblocks_far; bi++)
    {
        size_t i = F->block_far[2*bi];
        size_t j = F->block_far[2*bi+1];
        rank_map[i*F->nbcols+j] = M->far_rank[bi];
        if(i != j && F->symm == 'S')
            rank_map[j*F->nbcols+i] = M->far_rank[bi];
    }
    for(bi = 0; bi < F->nblocks_near; bi++)
    {
        size_t i = F->block_near[2*bi];
        size_t j = F->block_near[2*bi+1];
        int nrowsi = R->size[i];
        int ncolsj = C->size[j];
        int rank = nrowsi < ncolsj ? nrowsi : ncolsj;
        rank_map[i*F->nbcols+j] = rank;
        if(i != j && F->symm == 'S')
            rank_map[j*F->nbcols+i] = rank;
    }
    FILE *fd = fopen(filename, "w");
    fprintf(fd, "%d %d\n", F->nbrows, F->nbcols);
    for(size_t i = 0; i < F->nbrows; i++)
    {
        for(size_t j = 0; j < F->nbcols; j++)
            fprintf(fd, " %d", rank_map[i*F->nbcols+j]);
        fprintf(fd, "\n");
    }
    fclose(fd);
    return 0;
}
