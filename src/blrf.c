#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <string.h>
#include "stars.h"
#include "stars-misc.h"
#include "cblas.h"
#include "lapacke.h"


STARS_BLRF *STARS_BLRF_init(STARS_Problem *problem, char symm,
        STARS_Cluster *row_cluster, STARS_Cluster *col_cluster,
        int nblocks_far, int *block_far, int nblocks_near, int *block_near,
        STARS_BLRF_Type type)
// Initialization of structure STARS_BLRF
// Parameters:
//   problem: pointer to a structure, holding all the information about problem
//   symm: 'S' if problem and division into blocks are both symmetric, 'N'
//     otherwise.
//   row_cluster: clusterization of rows into block rows.
//   col_cluster: clusterization of columns into block columns.
//   nblocks_far: number of admissible far-field blocks.
//   block_far: array of pairs of admissible far-filed block rows and block
//     columns. block_far[2*i] is an index of block row and block_far[2*i+1]
//     is an index of block column.
//   nblocks_near: number of admissible far-field blocks.
//   block_near: array of pairs of admissible near-filed block rows and block
//     columns. block_near[2*i] is an index of block row and block_near[2*i+1]
//     is an index of block column.
//   type: type of block low-rank format. Tiled with STARS_BLRF_Tiled or
//     hierarchical with STARS_BLRF_H or STARS_BLRF_HOLDR.
{
    int i, j, bi;
    int *size;
    STARS_BLRF *blrf = malloc(sizeof(*blrf));
    blrf->problem = problem;
    blrf->symm = symm;
    blrf->nblocks_far = nblocks_far;
    blrf->block_far = block_far;
    blrf->block_near = block_near;
    blrf->nblocks_near = nblocks_near;
    blrf->row_cluster = row_cluster;
    int nbrows = blrf->nbrows = row_cluster->nblocks;
    // Set far-field block columns for each block row in compressed format
    blrf->brow_far_start = malloc((nbrows+1)*sizeof(int));
    blrf->brow_far = malloc(nblocks_far*sizeof(int));
    size = malloc(nbrows*sizeof(int));
    for(i = 0; i < nbrows; i++)
        size[i] = 0;
    for(bi = 0; bi < nblocks_far; bi++)
        size[block_far[2*bi]]++;
    blrf->brow_far_start[0] = 0;
    for(i = 0; i < nbrows; i++)
        blrf->brow_far_start[i+1] = blrf->brow_far_start[i]+size[i];
    for(i = 0; i < nbrows; i++)
        size[i] = 0;
    for(bi = 0; bi < nblocks_far; bi++)
    {
        i = block_far[2*bi];
        j = blrf->brow_far_start[i]+size[i];
        blrf->brow_far[j] = bi;//block_far[2*bi+1];
        size[i]++;
    }
    // Set near-field block columns for each block row in compressed format
    blrf->brow_near_start = malloc((nbrows+1)*sizeof(int));
    blrf->brow_near = malloc(nblocks_near*sizeof(int));
    size = malloc(nbrows*sizeof(int));
    for(i = 0; i < nbrows; i++)
        size[i] = 0;
    for(bi = 0; bi < nblocks_near; bi++)
        size[block_near[2*bi]]++;
    blrf->brow_near_start[0] = 0;
    for(i = 0; i < nbrows; i++)
        blrf->brow_near_start[i+1] = blrf->brow_near_start[i]+size[i];
    for(i = 0; i < nbrows; i++)
        size[i] = 0;
    for(bi = 0; bi < nblocks_near; bi++)
    {
        i = block_near[2*bi];
        j = blrf->brow_near_start[i]+size[i];
        blrf->brow_near[j] = bi;//block_near[2*bi+1];
        size[i]++;
    }
    free(size);
    if(symm == 'N')
    {
        blrf->col_cluster = col_cluster;
        int nbcols = blrf->nbcols = col_cluster->nblocks;
        // Set far-field block rows for each block column in compressed format
        blrf->bcol_far_start = malloc((nbcols+1)*sizeof(int));
        blrf->bcol_far = malloc(nblocks_far*sizeof(int));
        size = malloc(nbcols*sizeof(int));
        for(i = 0; i < nbcols; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_far; bi++)
            size[block_far[2*bi]]++;
        blrf->bcol_far_start[0] = 0;
        for(i = 0; i < nbcols; i++)
            blrf->bcol_far_start[i+1] = blrf->bcol_far_start[i]+size[i];
        for(i = 0; i < nbcols; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_far; bi++)
        {
            i = block_far[2*bi];
            j = blrf->bcol_far_start[i]+size[i];
            blrf->bcol_far[j] = bi;//block_far[2*bi+1];
            size[i]++;
        }
        // Set near-field block rows for each block column in compressed format
        blrf->bcol_near_start = malloc((nbcols+1)*sizeof(int));
        blrf->bcol_near = malloc(nblocks_near*sizeof(int));
        size = malloc(nbcols*sizeof(int));
        for(i = 0; i < nbcols; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_near; bi++)
            size[block_near[2*bi]]++;
        blrf->bcol_near_start[0] = 0;
        for(i = 0; i < nbcols; i++)
            blrf->bcol_near_start[i+1] = blrf->bcol_near_start[i]+size[i];
        for(i = 0; i < nbcols; i++)
            size[i] = 0;
        for(bi = 0; bi < nblocks_near; bi++)
        {
            i = block_near[2*bi];
            j = blrf->bcol_near_start[i]+size[i];
            blrf->bcol_near[j] = bi;//block_near[2*bi+1];
            size[i]++;
        }
        free(size);
    }
    else
    {
        blrf->col_cluster = row_cluster;
        blrf->nbcols = row_cluster->nblocks;
        // Set far-field block rows for each block column in compressed format
        blrf->bcol_far_start = blrf->brow_far_start;
        blrf->bcol_far = blrf->brow_far;
        // Set near-field block rows for each block column in compressed format
        blrf->bcol_near_start = blrf->brow_near_start;
        blrf->bcol_near = blrf->brow_near;
    }
    blrf->type = type;
    return blrf;
}

void STARS_BLRF_free(STARS_BLRF *blrf)
// Free memory, used by block low rank format (partitioning of array into
// blocks)
{
    if(blrf == NULL)
    {
        fprintf(stderr, "STARS_BLRF instance is NOT initialized\n");
        return;
    }
    free(blrf->brow_far_start);
    free(blrf->brow_far);
    free(blrf->brow_near_start);
    free(blrf->brow_near);
    if(blrf->symm == 'N')
    {
        free(blrf->bcol_far_start);
        free(blrf->bcol_far);
        free(blrf->bcol_near_start);
        free(blrf->bcol_near);
    }
    free(blrf);
}

void STARS_BLRF_info(STARS_BLRF *blrf)
// Print short info on block partitioning
{
    if(blrf == NULL)
    {
        fprintf(stderr, "STARS_BLRF instance is NOT initialized\n");
        return;
    }
    printf("<STARS_BLRF at %p, '%c' symmetric, %d block rows, %d "
            "block columns, %d far-field blocks, %d near-field blocks>\n",
            blrf, blrf->symm, blrf->nbrows, blrf->nbcols, blrf->nblocks_far,
            blrf->nblocks_near);
}

void STARS_BLRF_print(STARS_BLRF *blrf)
// Print full info on block partitioning
{
    int i, j;
    if(blrf == NULL)
    {
        printf("STARS_BLRF instance is NOT initialized\n");
        return;
    }
    printf("<STARS_BLRF at %p, '%c' symmetric, %d block rows, %d "
            "block columns, %d far-field blocks, %d near-field blocks>\n",
            blrf, blrf->symm, blrf->nbrows, blrf->nbcols, blrf->nblocks_far,
            blrf->nblocks_near);
    // Printing info about far-field blocks
    for(i = 0; i < blrf->nbrows; i++)
    {
        if(blrf->brow_far_start[i+1] > blrf->brow_far_start[i])
            printf("Admissible far-field block columns for block row %d: %d",
                    i, blrf->brow_far[blrf->brow_far_start[i]]);
        for(j = blrf->brow_far_start[i]+1; j < blrf->brow_far_start[i+1]; j++)
        {
            printf(" %d", blrf->brow_far[j]);
        }
        if(blrf->brow_far_start[i+1] > blrf->brow_far_start[i])
            printf("\n");
    }
    // Printing info about near-field blocks
    for(i = 0; i < blrf->nbrows; i++)
    {
        if(blrf->brow_near_start[i+1] > blrf->brow_near_start[i])
            printf("Admissible near-field block columns for block row %d: %d",
                    i, blrf->brow_near[blrf->brow_near_start[i]]);
        for(j = blrf->brow_near_start[i]+1; j < blrf->brow_near_start[i+1];
                j++)
        {
            printf(" %d", blrf->brow_near[j]);
        }
        if(blrf->brow_near_start[i+1] > blrf->brow_near_start[i])
            printf("\n");
    }
}

STARS_BLRF *STARS_BLRF_init_tiled(STARS_Problem *problem, STARS_Cluster
        *row_cluster, STARS_Cluster *col_cluster, char symm)
// Create plain division into tiles/blocks using plain cluster trees for rows
// and columns without actual pivoting
{
    if(symm == 'S' && problem->symm == 'N')
    {
        fprintf(stderr, "Since problem is NOT symmetric, can not proceed with "
                "symmetric flag on in STARS_BLRF_plain\n");
        exit(1);
    }
    if(symm == 'S' && row_cluster != col_cluster)
    {
        fprintf(stderr, "Since problem is symmetric, clusters should be the "
                "same (both pointers should be equal)\n");
        exit(1);
    }
    int nbrows = row_cluster->nblocks, nbcols = col_cluster->nblocks;
    int i, j, k = 0, nblocks_far, *block_far;
    if(symm == 'N')
    {
        nblocks_far = nbrows*nbcols;
        block_far = malloc(2*nblocks_far*sizeof(int));
        for(i = 0; i < nbrows; i++)
            for(j = 0; j < nbcols; j++)
            {
                block_far[2*k] = i;
                block_far[2*k+1] = j;
                k++;
            }
    }
    else
    {
        nblocks_far = nbrows*(nbrows+1)/2;
        block_far = malloc(2*nblocks_far*sizeof(int));
        for(i = 0; i < nbrows; i++)
            for(j = 0; j <= i; j++)
            {
                block_far[2*k] = i;
                block_far[2*k+1] = j;
                k++;
            }
    }
    return STARS_BLRF_init(problem, symm, row_cluster, col_cluster,
            nblocks_far, block_far, 0, NULL, STARS_BLRF_Tiled);
}

void STARS_BLRF_getblock(STARS_BLRF *blrf, int i, int j, int *shape, void **D)
// PLEASE CLEAN MEMORY POINTER *D AFTER USE
{
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
    *D = malloc(problem->entry_size*nrows*ncols);
    (problem->kernel)(nrows, ncols, row_cluster->pivot+row_cluster->start[i],
            col_cluster->pivot+col_cluster->start[j], problem->row_data,
            problem->col_data, *D);
}

/*
int batched_lowrank_approximation(STARS_BLRFmatrix *mat, int count, int *id,
        int maxrank, double tol, void **UV, int *rank)
{
    int bi, i, j;
    STARS_BLRF *format = mat->format;
    block_kernel kernel = format->problem->kernel;
    int max_rows = 0, max_cols = 0;
    for(i = 0; i < format->nbrows; i++)
        if(format->ibrow_size[i] > max_rows)
            max_rows = format->ibrow_size[i];
    for(i = 0; i < format->nbcols; i++)
        if(format->ibcol_size[i] > max_cols)
            max_cols = format->ibcol_size[i];
    int mx = max_cols > max_rows ? max_cols : max_rows;
    int dtype_size = format->problem->dtype_size;
    int block_size = max_rows*max_cols*dtype_size;
    int tlwork = (4*mx+7)*mx;
    int lwork = tlwork*dtype_size;
    int liwork = 8*mx*dtype_size;
    void *block, *work, *iwork, *U, *S, *V;
    int S_dtype_size;
    if(format->problem->dtype == 's')
        S_dtype_size = sizeof(float);
    else if(format->problem->dtype == 'd')
        S_dtype_size = sizeof(double);
    else if(format->problem->dtype == 'c')
        S_dtype_size = sizeof(float);
    else
        S_dtype_size = sizeof(double);
    //omp_set_max_active_levels(2);
    //printf("count=%d\n", count);
    #pragma omp parallel shared(block, work, iwork, U, V, S) private(i, j, bi)
    {
        int nthreads;
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
            //printf("Total threads %d\n", nthreads);
            block = malloc(nthreads*block_size);
            work = malloc(nthreads*lwork);
            iwork = malloc(nthreads*liwork);
            U = malloc(nthreads*block_size);
            V = malloc(nthreads*block_size);
            S = malloc(nthreads*mx*S_dtype_size);
            //printf("block_size %d, S_size %d\n", block_size,
            //mx*S_dtype_size);
        }
        #pragma omp barrier
        int tid = omp_get_thread_num();
        void *tblock = block+block_size*tid;
        void *twork = work+lwork*tid;
        void *tiwork = iwork+liwork*tid;
        void *tU = U+block_size*tid;
        void *tV = V+block_size*tid;
        void *tS = S+mx*S_dtype_size*tid;
        int tinfo = 0, rows, cols, mn, trank, k, l, bid;
        //printf("Work in thread %d\n", tid);
        #pragma omp for
        for(bi = 0; bi < count; bi++)
        {
            bid = id[bi];
            i = mat->bindex[2*bid];
            j = mat->bindex[2*bid+1];
            rows = format->ibrow_size[i];
            cols = format->ibcol_size[j];
            mn = rows > cols ? cols : rows;
            //printf("%d %d %d\n", bi, rows, cols);
            kernel(rows, cols, format->row_pivot+
                    format->ibrow_start[i],
                    format->col_pivot+format->ibcol_start[j],
                    format->problem->row_data,
                    format->problem->col_data, tblock);
            tinfo = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S', rows, cols,
                    tblock, rows, tS, tU, rows, tV, cols, twork, tlwork,
                    tiwork);
            double Sthresh = 0., Scur = 0.;
            double *ptrS = tS, *ptr, *ptrV = tV;
            for(k = 0; k < mn; k++)
                Sthresh += ptrS[k]*ptrS[k];
            Sthresh *= tol*tol;
            trank = 1;
            for(k = mn-1; k >= 1; k--)
            {
                Scur += ptrS[k]*ptrS[k];
                if(Sthresh < Scur)
                {
                    trank = k+1;
                    break;
                }
            }
            if(2*trank < mn)
            {
                cblas_dcopy(rows*trank, tU, 1, UV[bi], 1);
                ptr = UV[bi]+sizeof(double)*rows*trank;
                for(k = 0; k < cols; k++)
                    for(l = 0; l < trank; l++)
                    {
                        ptr[k*trank+l] = ptrS[l]*ptrV[k*mn+l];
                    }
                rank[bi] = trank;
            }
            else
                rank[bi] = -1;
        }
        #pragma omp master
        {
            free(block);
            free(work);
            free(iwork);
            free(U);
            free(V);
            free(S);
        }
        //#pragma omp barrier
    }
    return 0;
}

int batched_get_block(STARS_BLRFmatrix *mat, int count, int *id, void **A)
{
    STARS_BLRF *format = mat->format;
    block_kernel kernel = format->problem->kernel;
    #pragma omp parallel
    {
        int bi, i, j, rows, cols, nthreads, tid, bid;
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
            //printf("Total threads %d\n", nthreads);
        }
        #pragma omp barrier
        tid = omp_get_thread_num();
        //printf("Work in thread %d\n", tid);
        #pragma omp for
        for(bi = 0; bi < count; bi++)
        {
            bid = id[bi];
            i = mat->bindex[2*bid];
            j = mat->bindex[2*bid+1];
            rows = format->ibrow_size[i];
            cols = format->ibcol_size[j];
            kernel(rows, cols, format->row_pivot+
                    format->ibrow_start[i],
                    format->col_pivot+format->ibcol_start[j],
                    format->problem->row_data,
                    format->problem->col_data, A[bi]);
        }
        //#pragma omp barrier
    }
    return 0;
}

STARS_BLRFmatrix *STARS_blrf_batched_algebraic_compress(STARS_BLRF *format,
        int maxrank, double tol)
{
    int i, j, bi, mn;
    char symm = format->symm;
    int num_blocks = format->nbrows*format->nbcols;
    int total_blocks = num_blocks;
    if(symm == 'S')
        num_blocks = (format->nbrows+1)*format->nbrows/2;
    //printf("X %d %d\n", format->nbrows, num_blocks);
    int *block_id = (int *)malloc(num_blocks*sizeof(int));
    int batched_id = 0;
    STARS_BLRFmatrix *mat = (STARS_BLRFmatrix *)malloc(sizeof(STARS_BLRFmatrix));
    mat->bcount = total_blocks;
    mat->format = format;
    mat->bindex = (int *)malloc(2*total_blocks*sizeof(int));
    mat->brank = (int *)malloc(total_blocks*sizeof(int));
    int *rank = (int *)malloc(num_blocks*sizeof(int));
    mat->U = (Array **)malloc(total_blocks*sizeof(Array *));
    mat->V = (Array **)malloc(total_blocks*sizeof(Array *));
    mat->A = (Array **)malloc(total_blocks*sizeof(Array *));
    for(i = 0; i < format->nbrows; i++)
        for(j = 0; j < format->nbcols; j++)
        {
            bi = i * format->nbcols + j;
            mat->bindex[2*bi] = i;
            mat->bindex[2*bi+1] = j;
            if(i < j && symm == 'S')
            {
                mat->U[bi] = NULL;
                mat->V[bi] = NULL;
                mat->A[bi] = NULL;
                mat->brank[bi] = -1;
                continue;
            }
            //printf("bid %d\n", batched_id);
            block_id[batched_id] = bi;
            batched_id++;
        }
    int rows = format->ibrow_size[0], cols = format->ibcol_size[0];
    int uv_size = (rows+cols)*maxrank*sizeof(double);
    void *UV_alloc = malloc(uv_size*num_blocks);
    mat->UV_alloc = UV_alloc;
    void **UV = malloc(num_blocks*sizeof(void *));
    for(bi = 0; bi < num_blocks; bi++)
        UV[bi] = UV_alloc+uv_size*bi;
    //printf("num_blocks=%d\n", num_blocks);
    batched_lowrank_approximation(mat, num_blocks, block_id, maxrank,
            tol, UV, rank);
    //printf("num_blocks=%d\n", num_blocks);
    int shape[2], bid;
    int num_fullrank = 0;
    for(bi = 0; bi < num_blocks; bi++)
    {
        bid = block_id[bi];
        mat->brank[bid] = rank[bi];
        mat->A[bid] = NULL;
        i = mat->bindex[2*bid];
        j = mat->bindex[2*bid+1];
        if(rank[bi] == -1)
        {
            mat->U[bid] = NULL;
            mat->V[bid] = NULL;
            block_id[num_fullrank] = bid;
            num_fullrank++;
        }
        else
        {
            shape[0] = format->ibrow_size[i];
            shape[1] = rank[bi];
            //if(rank[bi] > maxrank)
            //    printf("DDDDD\n");
            mat->U[bid] = Array_from_buffer(2, shape, 'd', 'F', UV[bi]);
            shape[0] = shape[1];
            shape[1] = format->ibcol_size[j];
            mat->V[bid] = Array_from_buffer(2, shape, 'd', 'F', UV[bi]+
                    mat->U[bid]->nbytes);
        }
    }
    //printf("33\n");
    free(UV);
    int a_size = rows*cols*(format->problem->dtype_size);
    //printf("a_size %d, %d\n", a_size, num_fullrank);
    void *A_alloc = malloc(num_fullrank*a_size);
    //printf("A_alloc %p\n", A_alloc);
    mat->A_alloc = A_alloc;
    void **A = malloc(num_fullrank*sizeof(void *));
    for(bi = 0; bi < num_fullrank; bi++)
    {
        bid = block_id[bi];
        A[bi] = A_alloc+a_size*bi;
        shape[0] = format->ibrow_size[mat->bindex[2*bid]];
        shape[1] = format->ibcol_size[mat->bindex[2*bid+1]];
        //printf("(%d,%d)\n", shape[0], shape[1]);
        mat->A[bid] = Array_from_buffer(2, shape, format->problem->dtype,
                'F', A[bi]);
    } 
    //printf("GOING DEEP\n");
    batched_get_block(mat, num_fullrank, block_id, A);
    return mat;
}

*/
