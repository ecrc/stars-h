#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "stars.h"
#include "cblas.h"
#include "lapacke.h"


STARS_BLRM *STARS_BLRM_init(STARS_BLRF *blrf, int *far_rank, Array **far_U,
        Array **far_V, Array **far_D, int onfly, Array **near_D, void *alloc_U,
        void *alloc_V, void *alloc_D, char alloc_type)
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
    blrm->alloc_U = alloc_U;
    blrm->alloc_V = alloc_V;
    blrm->alloc_D = alloc_D;
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
    int bi;
    free(blrm->far_rank);
    if(blrm->alloc_type == '1')
    {
        free(blrm->alloc_U);
        free(blrm->alloc_V);
        if(blrm->alloc_D != NULL)
            free(blrm->alloc_D);
        for(bi = 0; bi < blrf->nblocks_far; bi++)
        {
            if(blrm->far_U[bi] != NULL)
            {
                blrm->far_U[bi]->buffer = NULL;
                Array_free(blrm->far_U[bi]);
            }
            if(blrm->far_V[bi] != NULL)
            {
                blrm->far_V[bi]->buffer = NULL;
                Array_free(blrm->far_V[bi]);
            }
            if(blrm->onfly == 0 && blrm->far_D[bi] != NULL)
            {
                blrm->far_D[bi]->buffer = NULL;
                Array_free(blrm->far_D[bi]);
            }
        }
        free(blrm->far_U);
        free(blrm->far_V);
        if(blrm->onfly == 0)
        {
            for(bi = 0; bi < blrf->nblocks_near; bi++)
                if(blrm->near_D[bi] != NULL)
                    Array_free(blrm->near_D[bi]);
            free(blrm->far_D);
            free(blrm->near_D);
        }
    }
    else if(blrm->alloc_type == '2')
    {
        for(bi = 0; bi < blrf->nblocks_far; bi++)
        {
            if(blrm->far_U[bi] != NULL)
                Array_free(blrm->far_U[bi]);
            if(blrm->far_V[bi] != NULL)
                Array_free(blrm->far_V[bi]);
            if(blrm->onfly == 0 && blrm->far_D[bi] != NULL)
                Array_free(blrm->far_D[bi]);
        }
        free(blrm->far_U);
        free(blrm->far_V);
        if(blrm->onfly == 0)
        {
            for(bi = 0; bi < blrf->nblocks_near; bi++)
                if(blrm->near_D[bi] != NULL)
                    Array_free(blrm->near_D[bi]);
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
    printf("<STARS_BLRM at %p, %d onfly, allocation type '%c'>\n",
            blrm, blrm->onfly, blrm->alloc_type);
}

void STARS_BLRM_error(STARS_BLRM *blrm)
// Measure error of approximation by non-nested block low-rank matrix
{
    STARS_BLRF *blrf = blrm->blrf;
    STARS_Problem *problem = blrf->problem;
    int bi, ndim = problem->ndim;
    if(ndim != 2)
    {
        fprintf(stderr, "Currently only scalar kernels are supported\n");
        return;
    }
    STARS_Cluster *row_cluster = blrf->row_cluster;
    STARS_Cluster *col_cluster = blrf->col_cluster;
    int nblocks_far = blrf->nblocks_far;
    int nblocks_near = blrf->nblocks_near;
    double diff = 0., norm = 0., tmpnorm, tmpdiff, tmperr, maxerr = 0.;
    Array *block, *block2;
    char symm = blrf->symm;
    #pragma omp parallel for
    for(bi = 0; bi < nblocks_far; bi++)
    {
        int i = blrf->block_far[2*bi];
        int j = blrf->block_far[2*bi+1];
        int nrowsi = row_cluster->size[i];
        int ncolsj = col_cluster->size[j];
        int shape[2] = {nrowsi, ncolsj};
        Array *block = Array_new(2, shape, problem->dtype, 'F');
        (problem->kernel)(nrowsi, ncolsj, row_cluster->pivot+
                row_cluster->start[i], col_cluster->pivot+
                col_cluster->start[j], problem->row_data, problem->col_data,
                block->buffer);
        double tmpnorm = Array_norm(block);
        #pragma omp critical
        {
            norm += tmpnorm*tmpnorm;
            if(i != j && symm == 'S')
                norm += tmpnorm*tmpnorm;
        }
        if(blrm->far_U[bi] != NULL && blrm->far_V[bi] != NULL)
        {
            Array *block2 = Array_dot(blrm->far_U[bi], blrm->far_V[bi]);
            double tmpdiff = Array_diff(block, block2);
            double tmperr = tmpdiff/tmpnorm;
            Array_free(block2);
            #pragma omp critical
            {
                diff += tmpdiff*tmpdiff;
                if(i != j && symm == 'S')
                    diff += tmpdiff*tmpdiff;
                if(tmperr > maxerr)
                    maxerr = tmperr;
            }
        }
        Array_free(block);
    }
    if(blrm->onfly == 0)
        for(bi = 0; bi < nblocks_near; bi++)
        {
            int i = blrf->block_near[2*bi];
            int j = blrf->block_near[2*bi+1];
            tmpnorm = Array_norm(blrm->near_D[bi]);
            norm += tmpnorm*tmpnorm;
            if(i != j && symm == 'S')
                norm += tmpnorm*tmpnorm;
        }
    else
        for(bi = 0; bi < nblocks_near; bi++)
        {
            int i = blrf->block_near[2*bi];
            int j = blrf->block_near[2*bi+1];
            int nrowsi = row_cluster->size[i];
            int ncolsj = col_cluster->size[j];
            int shape[2] = {nrowsi, ncolsj};
            block = Array_new(2, shape, problem->dtype, 'F');
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
        if(onfly == 0)
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
        Array_SVD(block, &U, &S, &V);
        rank = SVD_get_rank(S, tol, 'F');
        Array_free(block);
        if(rank < mn/2)
        // If block is low-rank
        {
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
    double total_time = omp_get_wtime();
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
    int *far_rank = malloc(nblocks_far*sizeof(int));
    STARS_Cluster *row_cluster = blrf->row_cluster;
    STARS_Cluster *col_cluster = blrf->col_cluster;
    int nrowsi, ncolsj;
    #pragma omp parallel private(bi, i, j, nrowsi, ncolsj, mn, block, block2,\
            rank, ptr, ptrS, ptrV, k, l, U, S, V)
    {
        int *shape = malloc(ndim*sizeof(int));
        memcpy(shape, blrf->problem->shape, ndim*sizeof(int));
        #pragma omp for
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
            block = Array_new(ndim, shape, problem->dtype, 'F');
            (problem->kernel)(nrowsi, ncolsj, row_cluster->pivot+
                    row_cluster->start[i], col_cluster->pivot+
                    col_cluster->start[j], problem->row_data,
                    problem->col_data, block->buffer);
            block2 = Array_copy(block, 'N');
            //dmatrix_lr(rows, cols, block->buffer, tol, &rank, &U, &S, &V);
            Array_SVD(block, &U, &S, &V);
            rank = SVD_get_rank(S, tol, 'F');
            Array_free(block);
            if(rank < mn/2)
            // If block is low-rank
            {
                //if(maxrank > 0)
                    // If block is low-rank and maximum rank is upperbounded,
                    // then rank should be equal to maxrank
                //    rank = maxrank;
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
                Array_free(block2);
                far_rank[bi] = mn;
            }
            Array_free(U);
            Array_free(S);
            Array_free(V);
        }
        free(shape);
    }
    /*
    if(onfly == 0)
    #pragma omp parallel private(bi, i, j, nrowsi, ncolsj, mn, block)
    {
        int *shape = malloc(ndim*sizeof(int));
        memcpy(shape, blrf->problem->shape, ndim*sizeof(int));
        #pragma omp for
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
    }
    */
    printf("TOTAL TIME: %f\n", omp_get_wtime()-total_time);
    return STARS_BLRM_init(blrf, far_rank, far_U, far_V, NULL, 1, NULL,
            NULL, NULL, NULL, '2');
}

STARS_BLRM *STARS_blrf_tiled_compress_algebraic_svd_batched(STARS_BLRF *blrf,
        int maxrank, double tol, int onfly, size_t max_buffer_size)
{
    double total_time = omp_get_wtime();
    STARS_Problem *problem = blrf->problem;
    int bbi, bi, ndim = problem->ndim, mn, rank;
    if(ndim != 2)
    {
        fprintf(stderr, "Currently only scalar kernels are supported\n");
        return NULL;
    }
    char symm = blrf->symm;
    int nblocks_far = blrf->nblocks_far;
    int nblocks_near = blrf->nblocks_near;
    Array **far_U = malloc(nblocks_far*sizeof(Array *));
    Array **far_V = malloc(nblocks_far*sizeof(Array *));
    int *shape = problem->shape;
    void *alloc_U = malloc(2*blrf->nbcols*shape[0]*maxrank*sizeof(double));
    void *alloc_V = malloc(2*blrf->nbrows*shape[1]*maxrank*sizeof(double));
    void *current_U = alloc_U, *current_V = alloc_V;
    STARS_Cluster *row_cluster = blrf->row_cluster;
    STARS_Cluster *col_cluster = blrf->col_cluster;
    int nrowsi, ncolsj;
    int *far_rank = malloc(nblocks_far*sizeof(int));
    void *row_data = problem->row_data;
    void *col_data = problem->col_data;
    size_t *ltotalwork = malloc(nblocks_far*sizeof(size_t));
    int *lwork_arrays = malloc(5*nblocks_far*sizeof(int));
    int *lbwork = lwork_arrays;
    int *luvwork = lwork_arrays+nblocks_far;
    int *lwork = lwork_arrays+2*nblocks_far;
    int *lswork = lwork_arrays+3*nblocks_far;
    int *liwork = lwork_arrays+4*nblocks_far;
    void *tmp_buffer = malloc(max_buffer_size);
    double tmp_time = omp_get_wtime(), tmp_time2;
    #pragma omp parallel for
    for(bi = 0; bi < nblocks_far; bi++)
    {
        int i = blrf->block_far[2*bi];
        int j = blrf->block_far[2*bi+1];
        int nrowsi = row_cluster->size[i];
        int ncolsj = col_cluster->size[j];
        if(nrowsi < ncolsj)
        {
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
    int nblocks_processed = 0;
    while(nblocks_processed < nblocks_far)
    {
        //printf("%d %d\n", nblocks_processed, nblocks_far);
        size_t tmp_ltotalwork = 0;
        int tmp_lbwork = 0, tmp_luvwork = 0, tmp_lwork = 0, tmp_lswork = 0;
        bi = nblocks_processed;
        while(bi < nblocks_far && tmp_ltotalwork+
                ltotalwork[bi] < max_buffer_size)
        {
            tmp_ltotalwork += ltotalwork[bi];
            tmp_lbwork += lbwork[bi];
            tmp_luvwork += luvwork[bi];
            tmp_lwork += lwork[bi];
            tmp_lswork += lswork[bi];
            bi++;
        }
        int batch_size = bi-nblocks_processed;
        int nrows[batch_size], ncols[batch_size];
        int *irow[batch_size], *icol[batch_size];
        double *buffer[batch_size];
        double *U[batch_size], *S[batch_size], *V[batch_size], *UV[batch_size];
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
        #pragma omp parallel for
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            int bi = bbi+nblocks_processed;
            int i = blrf->block_far[2*bi];
            int j = blrf->block_far[2*bi+1];
            nrows[bbi] = row_cluster->size[i];
            ncols[bbi] = col_cluster->size[j];
            irow[bbi] = row_cluster->pivot+row_cluster->start[i];
            icol[bbi] = col_cluster->pivot+col_cluster->start[j];
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
        #pragma omp parallel for
        for(bbi = 0; bbi < batch_size; bbi++)
            problem->kernel(nrows[bbi], ncols[bbi], irow[bbi], icol[bbi],
                    row_data, col_data, buffer[bbi]);
        //tmp_time2 = omp_get_wtime();
        //printf("TIME4: %f\n", tmp_time2-tmp_time);
        //tmp_time = tmp_time2;
        //printf("DONE WITH KERNEL\n");
        #pragma omp parallel for
        for(bbi = 0; bbi < batch_size; bbi++)
            //LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, jobu[bbi], jobv[bbi],
            //        nrows[bbi], ncols[bbi], buffer[bbi], nrows[bbi], S[bbi],
            //        U[bbi], nrows[bbi], V[bbi], ldv[bbi], work[bbi],
            //        lwork[bbi]);
            LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'O', nrows[bbi], ncols[bbi],
                    buffer[bbi], nrows[bbi], S[bbi], U[bbi], nrows[bbi],
                    V[bbi], ldv[bbi], work[bbi], lwork[bbi+nblocks_processed],
                    iwork[bbi]);
        //printf("DONE WITH SVD\n");
        //tmp_time2 = omp_get_wtime();
        //printf("TIME5: %f\n", tmp_time2-tmp_time);
        //tmp_time = tmp_time2;
        #pragma omp parallel for
        for(bbi = 0; bbi < batch_size; bbi++)
        {
            int bi = bbi+nblocks_processed;
            double *ptrS = S[bbi];
            double Stol = 0, Stmp = 0.;
            int i, j, mn = ldv[bbi], rank = mn;
            int shapeU[2] = {nrows[bbi], 0}, shapeV[2] = {0, ncols[bbi]};
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
                #pragma omp critical
                {
                    far_U[bi] = Array_from_buffer(2, shapeU, 'd', 'F',
                            current_U);
                    current_U += 2*shapeU[0]*shapeU[1]*sizeof(double);
                    far_V[bi] = Array_from_buffer(2, shapeV, 'd', 'F',
                            current_V);
                    current_V += 2*shapeV[0]*shapeV[1]*sizeof(double);
                }
                cblas_dcopy(shapeU[0]*shapeU[1], U[bbi], 1, far_U[bi]->buffer,
                        1);
                double *ptr = far_V[bi]->buffer, *ptrV = V[bbi];
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
    return STARS_BLRM_init(blrf, far_rank, far_U, far_V, NULL, 1, NULL,
            alloc_U, alloc_V, NULL, '1');
}
