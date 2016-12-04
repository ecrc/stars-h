#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include "stars.h"
//#include "cblas.h"
//#include "lapacke.h"
#include "misc.h"


int STARS_BLRM_new(STARS_BLRM **M, STARS_BLRF *F, int *far_rank,
        Array **far_U, Array **far_V, int onfly,
        Array **near_D, void *alloc_U, void *alloc_V,
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
    size_t bi, data_size = 0, size = 0;
    size += sizeof(*M2);
    size += F->nblocks_far*(sizeof(*far_rank)+sizeof(*far_U)+sizeof(*far_V));
    size += F->nblocks_near*sizeof(*near_D);
    for(bi = 0; bi < F->nblocks_far; bi++)
    {
        size += far_U[bi]->nbytes+far_V[bi]->nbytes;
        data_size += far_U[bi]->data_nbytes+far_V[bi]->data_nbytes;
    }
    for(bi = 0; bi < F->nblocks_near; bi++)
    {
        size += near_D[bi]->nbytes;
        data_size += near_D[bi]->data_nbytes;
    }
    M2->nbytes = size;
    M2->data_nbytes = data_size;
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
    printf("<STARS_BLRM at %p, %d onfly, allocation type '%c', %f MB memory "
            "footprint>\n", M, M->onfly, M->alloc_type, M->nbytes/1024./1024.);
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
    size_t bi = -1, k;
    if(F->nblocks_far > 0)
    {
        k = F->brow_far_start[i];
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
            *U = M->far_U[bi]->data;
            *V = M->far_V[bi]->data;
            return info;
        }
    }
    if(F->nblocks_near > 0)
    {
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
            *D = M->near_D[bi]->data;
            if(*D == NULL)
                info = STARS_BLRF_get_block(F, i, j, shape, D);
            return info;
        }
    }
    STARS_WARNING("Required block (%d, %d) is not admissible!\n", i, j);
    info = STARS_BLRF_get_block(F, i, j, shape, D);
    return info;
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
    int info = Array_new(A, P->ndim, P->shape, P->dtype, 'F');
    Array *A2 = *A;
    double *ptrA = A2->data;
    int lda = A2->shape[0];
    size_t bi;
    volatile int abort = 0;
    // At first restore far-field blocks
    #pragma omp parallel for
    for(bi = 0; bi < F->nblocks_far; bi++)
    {
        if(abort != 0)
            continue;
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        Array *B;
        int info = Array_dot(M->far_U[bi], M->far_V[bi], &B);
        int shape[2], rank;
        void *U, *V, *D;
        info = STARS_BLRM_get_block(M, i, j, shape, &rank, &U, &V, &D);
        if(U != M->far_U[bi]->data || V != M->far_V[bi]->data || D != NULL)
        {
            STARS_ERROR("bad BLRM_get_block()");
            info = 2;
        }
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
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
    if(abort != 0)
        return abort;
    // Restore near-field blocks
    #pragma omp parallel for
    for(bi = 0; bi < F->nblocks_near; bi++)
    {
        if(abort != 0)
            continue;
        int i = F->block_near[2*bi];
        int j = F->block_near[2*bi+1];
        Array *B = M->near_D[bi];
        int shape[2], rank;
        void *U, *V, *D;
        int info = STARS_BLRM_get_block(M, i, j, shape, &rank, &U, &V, &D);
        if(U != NULL || V != NULL || (F->symm == 0 && D != B->data))
        {
            STARS_ERROR("bad BLRM_get_block()");
            info = 2;
        }
        if(info != 0)
        {
            #pragma omp atomic write
            abort = info;
            continue;
        }
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
    return abort;
}

int STARS_BLRM_tiled_compress_algebraic_svd(STARS_BLRM **M, STARS_BLRF *F,
        int fixrank, double tol, int onfly)
// Uses SVD to acquire rank of each block, compresses given matrix (given
// by block kernel, which returns submatrices) with relative accuracy tol
{
    //Set timer
    double tmp_time = omp_get_wtime(), tmp_time2;
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
    tmp_time2 = omp_get_wtime();
    STARS_WARNING("total time: %f sec", tmp_time2-tmp_time);
    // Init Block Low-Rank Matrix with given buffers
    return STARS_BLRM_new(M, F, far_rank, far_U, far_V, onfly, near_D,
            NULL, NULL, NULL, '2');
}

int STARS_BLRM_tiled_compress_algebraic_svd_ompfor(STARS_BLRM **M,
        STARS_BLRF *F, int fixrank, double tol, int onfly, int nthreads_outer,
        int nthreads_inner)
// Uses SVD to acquire rank of each block, compresses given matrix (given
// by block kernel, which returns submatrices) with relative accuracy tol
{
    // Set timer
    double tmp_time = omp_get_wtime(), tmp_time2;
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
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near;
    // Following values default to given block low-rank format F, but they are
    // changed when there are false far-field blocks.
    size_t new_nblocks_far = nblocks_far, new_nblocks_near = nblocks_near;
    int *block_far = F->block_far, *block_near = F->block_near;
    // Places to store low-rank factors, dense blocks and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank = NULL;
    // Init buffers to store low-rank factors of far-field blocks if needed
    if(nblocks_far > 0)
    {
        STARS_MALLOC(far_U, nblocks_far);
        STARS_MALLOC(far_V, nblocks_far);
        STARS_MALLOC(far_rank, nblocks_far);
    }
    // Shortcuts to information about clusters
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    // Work variables
    int info;
    size_t bi, bj = 0;
    volatile int abort = 0;
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    omp_set_dynamic(0);
    mkl_set_dynamic(0);
    omp_set_num_threads(nthreads_inner);
    mkl_set_num_threads(nthreads_inner);
    #pragma omp parallel for num_threads(nthreads_outer)
    for(bi = 0; bi < nblocks_far; bi++)
    // Cycle over every admissible block
    {
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
        int rank = fixrank;
        Array *A, *A2, *U, *S, *V;
        // Get array, holding necessary data
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
        // Create copy for SVD
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
        // Get rank
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
            if(fixrank != 0)
                rank = fixrank;
            far_rank[bi] = rank;
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
            // Copy part of `U` into low-rank factor `far_U`
            cblas_dcopy(rank*nrowsi, U->data, 1, far_U[bi]->data, 1);
            double *ptr = far_V[bi]->data;
            double *ptrS = S->data;
            double *ptrV = V->data;
            // Copy part of `S`*`V` into low-rank factor `far_V`
            for(size_t k = 0; k < ncolsj; k++)
                for(size_t l = 0; l < rank; l++)
                {
                    ptr[k*rank+l] = ptrS[l]*ptrV[k*mn+l];
                }
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
        // Free temporary SVD buffers
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
    omp_set_num_threads(nthreads_outer);
    if(abort != 0)
        return abort;
    // Get number of false far-field blocks
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
        // Update list of far-field blocks
        new_nblocks_far = nblocks_far-nblocks_false_far;
        if(new_nblocks_far > 0)
        {
            STARS_MALLOC(block_far, 2*new_nblocks_far);
            bj = 0;
            for(bi = 0; bi < nblocks_far; bi++)
            {
                // `false_far` must be in ascending order for this to work
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
        // Update format be creating new format and swapping buffers with it
        STARS_BLRF *F2;
        info = STARS_BLRF_new(&F2, P, F->symm, R, C, new_nblocks_far,
                block_far, new_nblocks_near, block_near, F->type);
        // Swap internal data of formats and free unnecessary data
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
        // And then with false far-field blockso
        #pragma omp parallel for
        for(bi = nblocks_near; bi < new_nblocks_near; bi++)
        {
            size_t bj = false_far[bi-nblocks_near];
            near_D[bi] = far_U[bj];
        }
    }
    // Changing size of far_rank, far_U and far_V
    if(nblocks_false_far > 0 && new_nblocks_far > 0)
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
    // Dealloc list of false far-field blocks if it is not empty
    if(nblocks_false_far > 0)
        free(false_far);
    // Finish with creating instance of Block Low-Rank Matrix with given
    // buffers
    tmp_time2 = omp_get_wtime();
    STARS_WARNING("total time: %f sec", tmp_time2-tmp_time);
    return STARS_BLRM_new(M, F, far_rank, far_U, far_V, onfly, near_D,
            NULL, NULL, NULL, '2');
}


int STARS_BLRM_tiled_compress_algebraic_svd_batched(STARS_BLRM **M,
        STARS_BLRF *F, int fixrank, double tol, int onfly, int maxrank,
        size_t max_buffer_size, int nthreads_outer, int nthreads_inner)
// The same approximation but with preparations for batched kernels
{
    // Set timer
    double tmp_time = omp_get_wtime(), tmp_time2;
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
    if(maxrank < 0)
    {
        STARS_ERROR("invalid value of `maxrank`");
        return 1;
    }
    if(max_buffer_size == 0)
    {
        STARS_ERROR("invalid value of `max_buffer_size`");
        return 1;
    }
    STARS_Problem *P = F->problem;
    int ndim = P->ndim;
    if(ndim != 2)
    {
        STARS_ERROR("only scalar kernels are supported");
        return 1;
    }
    // If rank is set by fixrank, then make maxrank equal to fixrank
    if(fixrank != 0)
        maxrank = fixrank;
    // bi for block index, bbi for batched block index
    size_t bbi, bi;
    size_t nblocks_far = F->nblocks_far, nblocks_near = F->nblocks_near;
    // Set variables for possibly required update of `F`
    size_t new_nblocks_far = nblocks_far, new_nblocks_near = nblocks_near;
    int *block_far = F->block_far, *block_near = F->block_near;
    // Places to store far-field blocks' low-rank factors and ranks
    Array **far_U = NULL, **far_V = NULL, **near_D = NULL;
    int *far_rank;
    STARS_MALLOC(far_U, nblocks_far);
    STARS_MALLOC(far_V, nblocks_far);
    STARS_MALLOC(far_rank, nblocks_far);
    int *shape = P->shape;
    // Compute maximum memory requirement for all low-rank factors
    size_t alloc_U_size = (size_t)F->nbcols*(size_t)shape[0]*
            (size_t)maxrank*sizeof(double);
    size_t alloc_V_size = (size_t)F->nbrows*(size_t)shape[1]*
            (size_t)maxrank*sizeof(double);
    size_t alloc_D_step = (size_t)shape[0]/F->nbrows*(size_t)shape[1]*
            sizeof(double);
    size_t alloc_D_size = alloc_D_step;
    void *alloc_U, *alloc_V;
    void *alloc_D;
    STARS_MALLOC(alloc_U, alloc_U_size);
    STARS_MALLOC(alloc_V, alloc_V_size);
    STARS_MALLOC(alloc_D, alloc_D_size);
    // Special variables to store pointers to current free memory in big
    // buffers
    void *current_U = alloc_U, *current_V = alloc_V;
    size_t current_D = 0;
    // Shortcuts for information about clusterization
    STARS_Cluster *R = F->row_cluster, *C = F->col_cluster;
    void *row_data = P->row_data, *col_data = P->col_data;
    // Sizes of temporary buffers for each block to make possible to build
    // approximation in kernel-centric (batched) mode
    size_t *ltotalwork, *lwork_arrays;
    STARS_MALLOC(ltotalwork, nblocks_far);
    STARS_MALLOC(lwork_arrays, 5*nblocks_far);
    size_t *lbwork = lwork_arrays;
    size_t *luvwork = lwork_arrays+nblocks_far;
    size_t *lwork = lwork_arrays+2*nblocks_far;
    size_t *lswork = lwork_arrays+3*nblocks_far;
    size_t *liwork = lwork_arrays+4*nblocks_far;
    // Temporary buffer for batched operations
    void *tmp_buffer;
    STARS_MALLOC(tmp_buffer, max_buffer_size);
    volatile int abort = 0;
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    omp_set_dynamic(0);
    mkl_set_dynamic(0);
    omp_set_num_threads(nthreads_inner);
    mkl_set_num_threads(nthreads_inner);
    #pragma omp parallel for num_threads(nthreads_outer)
    for(bi = 0; bi < nblocks_far; bi++)
    // Cycle over each far-field block
    {
        int i = F->block_far[2*bi];
        int j = F->block_far[2*bi+1];
        size_t nrowsi = R->size[i], ncolsj = C->size[j];
        // Fill in sizes of temporary buffers for SVD (GESDD)
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
            ltotalwork[bi] = (2*lbwork[bi]+luvwork[bi]+lwork[bi]+lswork[bi])*
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
            ltotalwork[bi] = (2*lbwork[bi]+luvwork[bi]+lwork[bi]+lswork[bi])*
                    sizeof(double)+liwork[bi]*sizeof(int);
        }
    }
    // Number of already processed blocks
    size_t nblocks_processed = 0;
    while(nblocks_processed < nblocks_far)
    {
        // Get number of blocks for next batch
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
        // Prepare all parameters for matrix kernel and SVD
        int nrows[batch_size], ncols[batch_size], *irow[batch_size],
            *icol[batch_size], ldv[batch_size], *iwork[batch_size];
        double *buffer[batch_size], *buffer_copy[batch_size], *U[batch_size],
               *S[batch_size], *V[batch_size], *UV[batch_size],
               *work[batch_size];
        // At first get pointers to temporary buffers for all matrices
        buffer[0] = tmp_buffer;
        buffer_copy[0] = buffer[0]+tmp_lbwork;
        UV[0] = buffer_copy[0]+tmp_lbwork;
        S[0] = UV[0]+tmp_luvwork;
        work[0] = S[0]+tmp_lswork;
        iwork[0] = (int *)(work[0]+tmp_lwork);
        for(bbi = 0; bbi < batch_size-1; bbi++)
        {
            buffer[bbi+1] = buffer[bbi]+lbwork[bbi+nblocks_processed];
            buffer_copy[bbi+1] = buffer[bbi+1]+tmp_lbwork;
            UV[bbi+1] = UV[bbi]+luvwork[bbi+nblocks_processed];
            S[bbi+1] = S[bbi]+lswork[bbi+nblocks_processed];
            work[bbi+1] = work[bbi]+lwork[bbi+nblocks_processed];
            iwork[bbi+1] = iwork[bbi]+liwork[bbi+nblocks_processed];
        }
        // Fill all other needed parameters for matrix kernel and SVD
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
        // Kernel-centric call for matrix kernel
        #pragma omp parallel for num_threads(nthreads_outer)
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            if(abort != 0)
                continue;
            int info = P->kernel(nrows[bbi], ncols[bbi], irow[bbi], icol[bbi],
                    row_data, col_data, buffer[bbi]);
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
        }
        if(abort != 0)
            return abort;
        // Kernel-centric copying buffers, which are useful if some far-field
        // blocks are actually near-field (so-called false far-field blocks)
        // If number of elements in some block does not fit into int type,
        // there will be problems
        #pragma omp parallel for num_threads(nthreads_outer)
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            cblas_dcopy(nrows[bbi]*ncols[bbi], buffer[bbi], 1,
                    buffer_copy[bbi], 1);
        }
        // Kernel-centric call for SVD (GESDD)
        #pragma omp parallel for num_threads(nthreads_outer)
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            if(abort != 0)
                continue;
            int info = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'O', nrows[bbi],
                    ncols[bbi], buffer[bbi], nrows[bbi], S[bbi], U[bbi],
                    nrows[bbi], V[bbi], ldv[bbi], work[bbi],
                    lwork[bbi+nblocks_processed], iwork[bbi]);
            if(info != 0)
            {
                #pragma omp atomic write
                abort = info;
                continue;
            }
        }
        if(abort != 0)
            return abort;
        // Compute ranks. Rank is set to -1 for false far-field blocks (which
        // appear to be dense instead of low-rank)
        #pragma omp parallel for num_threads(nthreads_outer)
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            if(abort != 0)
                continue;
            size_t bi = bbi+nblocks_processed;
            double *ptrS = S[bbi];
            size_t i, j, mn = ldv[bbi], rank = fixrank;
            int shapeU[2] = {nrows[bbi], 0}, shapeV[2] = {0, ncols[bbi]}, info;
            double Stol = 0, Stmp = ptrS[mn-1]*ptrS[mn-1];
            rank = mn;
            for(i = 0; i < mn; i++)
                Stol += ptrS[i]*ptrS[i];
            // If total norm is 0, then rank is 0, otherwise rank is > 0
            if(Stol == 0)
                rank = 0;
            Stol *= tol*tol;
            while(rank > 1 && Stol >= Stmp)
            {
                rank--;
                Stmp += ptrS[rank-1]*ptrS[rank-1];
            }
            if(rank < mn/2)
            // If block is low-rank
            {
                if(fixrank != 0)
                    rank = fixrank;
                else if(rank > maxrank)
                    rank = maxrank;
                far_rank[bi] = rank;
                shapeU[1] = rank;
                shapeV[0] = rank;
                #pragma omp critical
                {
                    info = Array_from_buffer(far_U+bi, 2, shapeU, 'd', 'F',
                            current_U);
                    if(info != 0)
                    {
                        #pragma omp atomic write
                        abort = info;
                    }
                    current_U += (size_t)shapeU[0]*(size_t)shapeU[1]*
                            sizeof(double);
                    info = Array_from_buffer(far_V+bi, 2, shapeV, 'd', 'F',
                            current_V);
                    if(info != 0)
                    {
                        #pragma omp atomic write
                        abort = info;
                    }
                    current_V += (size_t)shapeV[0]*(size_t)shapeV[1]*
                            sizeof(double);
                }
                if(abort != 0)
                    continue;
                // THIS MAY LEAD TO PROBLEM !! BETTER TO DO CYCLE OF COPIES
                cblas_dcopy((size_t)shapeU[0]*(size_t)shapeU[1], U[bbi], 1,
                        far_U[bi]->data, 1);
                double *ptr = far_V[bi]->data, *ptrV = V[bbi];
                for(i = 0; i < shapeV[1]; i++)
                    for(j = 0; j < shapeV[0]; j++)
                        ptr[i*rank+j] = ptrS[j]*ptrV[i*mn+j];
            }
            else
            // If block is dense
            #pragma omp critical
            {
                far_rank[bi] = -1;
                shapeU[1] = shapeV[1];
                // Use far_U[bi] to move data
                size_t sizeU = (size_t)shapeU[0]*(size_t)shapeU[1]*
                        sizeof(double);
                while(current_D+sizeU > alloc_D_size)
                {
                    alloc_D_size += alloc_D_step;
                    STARS_PREALLOC(alloc_D, alloc_D_size, abort);
                }
                info = Array_from_buffer(far_U+bi, 2, shapeU, 'd', 'F',
                        alloc_D+current_D);
                if(info != 0)
                {
                    #pragma omp atomic write
                    abort = info;
                }
                // Store real offset in far_V[bi], since only offset matters
                // with possibly changing base alloc_D
                far_V[bi] = alloc_D;
                current_D += sizeU;
                cblas_dcopy(shapeU[0]*shapeU[1], buffer_copy[bbi], 1,
                        far_U[bi]->data, 1);
            }
        }
        if(abort != 0)
            return abort;
        // Update number of processed far-field blocks
        nblocks_processed += batch_size;
    }
    omp_set_num_threads(nthreads_outer);
    // Get number of false far-field blocks
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
        size_t bj = 0;
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
        // Update list of far-field blocks
        new_nblocks_far = nblocks_far-nblocks_false_far;
        if(new_nblocks_far > 0)
        {
            STARS_MALLOC(block_far, 2*new_nblocks_far);
            size_t bj = 0;
            for(bi = 0; bi < nblocks_far; bi++)
            {
                // `false_far` must be in ascending order for this to work
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
        // Update format be creating new format and swapping buffers with it
        STARS_BLRF *F2;
        int info = STARS_BLRF_new(&F2, P, F->symm, R, C, new_nblocks_far,
                block_far, new_nblocks_near, block_near, F->type);
        // Swap internal data of formats and free unnecessary data
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
        // And then with false far-field blockso
        #pragma omp parallel for
        for(bi = nblocks_near; bi < new_nblocks_near; bi++)
        {
            size_t bj = false_far[bi-nblocks_near];
            near_D[bi] = far_U[bj];
            near_D[bi]->data = (far_U[bj]->data-(void *)far_V[bj])+alloc_D;
        }
    }
    // Changing size of far_rank, far_U and far_V
    if(nblocks_false_far > 0 && new_nblocks_far > 0)
    {
        size_t bj = 0;
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
    // Dealloc list of false far-field blocks if it is not empty
    if(nblocks_false_far > 0)
        free(false_far);
    // Free temporary buffers
    free(tmp_buffer);
    free(ltotalwork);
    free(lwork_arrays);
    tmp_time2 = omp_get_wtime()-tmp_time;
    STARS_WARNING("total time: %f", tmp_time2);
    return STARS_BLRM_new(M, F, far_rank, far_U, far_V, onfly, near_D, alloc_U,
            alloc_V, alloc_D, '1');
}
