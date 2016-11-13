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


STARS_BLR *STARS_BLR_init(STARS_Problem *problem, char symm, int *row_pivot,
        int *col_pivot, int nbrows, int nbcols, int *ibrow_start,
        int *ibcol_start, int *ibrow_size, int *ibcol_size,
        int admissible_nblocks, int *admissible_block_start,
        int *admissible_block_size, int *admissible_block,
        STARS_BlockStatus *admissible_block_status)
// Initialization of structure STARS_BLR
// Parameters:
//   problem: pointer to a structure, holding all the information about problem
//   symm: 'S' if problem and division into blocks are both symmetric, 'N'
//     otherwise
//   row_pivot: pivoting for rows, such that rows of each block are placed one
//     after another
//   col_pivot: pivoting for columns, such that columns of each block are
//     placed one after another
//   nbrows: number of block rows
//   nbcols: number of block columns
//   ibrow_start: array of start rows for each block row
//   ibcol_start: array of start column for each block column
//   ibrow_size: array of numbers of rows, presented in each block row
//   ibcol_size: array of numbers of columns, presented in each block column
{
    int i;
    STARS_BLR *blr = (STARS_BLR *)malloc(sizeof(STARS_BLR));
    blr->problem = problem;
    blr->symm = symm;
    blr->nrows = problem->shape[0];
    blr->ncols = problem->shape[problem->ndim-1];
    blr->row_pivot = (int *)malloc(blr->nrows*sizeof(int));
    if(row_pivot == NULL)
        for(i = 0; i < blr->nrows; i++)
            blr->row_pivot[i] = i;
    else
        memcpy(blr->row_pivot, row_pivot, blr->nrows*sizeof(int));
    blr->col_pivot = (int *)malloc(blr->ncols*sizeof(int));
    if(col_pivot == NULL)
        for(i = 0; i < blr->ncols; i++)
            blr->col_pivot[i] = i;
    else
        memcpy(blr->col_pivot, col_pivot, blr->ncols*sizeof(int));
    blr->nbrows = nbrows;
    blr->nbcols = nbcols;
    blr->nblocks = nbrows*nbcols;
    blr->ibrow_start = (int *)malloc(nbrows*sizeof(int));
    memcpy(blr->ibrow_start, ibrow_start, blr->nrows*sizeof(int));
    blr->ibrow_size = (int *)malloc(blr->nrows*sizeof(int));
    memcpy(blr->ibrow_size, ibrow_size, blr->nrows*sizeof(int));
    if(symm == 'N')
    {
        blr->ibcol_start = (int *)malloc(blr->ncols*sizeof(int));
        memcpy(blr->ibcol_start, ibcol_start, blr->ncols*sizeof(int));
        blr->ibcol_size = (int *)malloc(blr->ncols*sizeof(int));
        memcpy(blr->ibcol_size, ibcol_size, blr->ncols*sizeof(int));
    }
    else
    {
        blr->ibcol_start = blr->ibrow_start;
        blr->ibcol_size = blr->ibrow_size;
    }
    blr->admissible_nblocks = admissible_nblocks;
    blr->admissible_block_start = (int *)malloc(blr->nblocks*sizeof(int));
    memcpy(blr->admissible_block_start, admissible_block_start, blr->nblocks*
            sizeof(int));
    blr->admissible_block_size = (int *)malloc(blr->nblocks*sizeof(int));
    memcpy(blr->admissible_block_size, admissible_block_size, blr->nblocks*
            sizeof(int));
    blr->admissible_block = (int *)malloc(admissible_nblocks*sizeof(int));
    memcpy(blr->admissible_block, admissible_block, admissible_nblocks*
            sizeof(int));
    blr->admissible_block_status = (STARS_BlockStatus *)malloc(
            admissible_nblocks*sizeof(STARS_BlockStatus));
    memcpy(blr->admissible_block_status, admissible_block_status,
            admissible_nblocks*sizeof(STARS_BlockStatus));
    return blr;
}

void STARS_BLR_free(STARS_BLR *blr)
// Free memory, used by block low rank format (partitioning of array into
// blocks)
{
    if(blr == NULL)
    {
        fprintf(stderr, "STARS_BLR instance is NOT initialized\n");
        return;
    }
    free(blr->row_pivot);
    free(blr->ibrow_start);
    free(blr->ibrow_size);
    free(blr->admissible_block_start);
    free(blr->admissible_block_size);
    free(blr->admissible_block);
    free(blr->admissible_block_status);
    if(blr->symm == 'N')
    {
       free(blr->col_pivot);
       free(blr->ibcol_start);
       free(blr->ibcol_size);
    }
    free(blr);
}

void STARS_BLR_info(STARS_BLR *blr)
// Print short info on block partitioning
{
    int i, j, k;
    if(blr == NULL)
    {
        printf("STARS_BLR NOT initialized\n");
        return;
    }
    if(blr->symm == 'S')
        printf("Symmetric partitioning into blocks\n(blocking for columns "
                "is the same, as blocking for rows)\n");
    printf("%d block rows, ", blr->nbrows);
    if(blr->symm == 'N')
        printf("%d block columns, ", blr->nbcols);
    printf("%d admissible blocks\n", blr->admissible_nblocks);
}
void STARS_BLR_print(STARS_BLR *blr)
// Print full info on block partitioning
{
    int i, j, k;
    if(blr == NULL)
    {
        printf("STARS_BLR NOT initialized\n");
        return;
    }
    if(blr->symm == 'S')
        printf("Symmetric partitioning into blocks\n(blocking for columns "
                "is the same, as blocking for rows)\n");
    printf("%d block rows (start, end):\n", blr->nbrows);
    i = 0;
    if(blr->nbrows > 0)
        printf("(%i, %i)", blr->ibrow_start[i],
                blr->ibrow_start[i]+blr->ibrow_size[i]);
    for(i = 1; i < blr->nbrows; i++)
    {
        printf(", (%i, %i)", blr->ibrow_start[i],
                blr->ibrow_start[i]+blr->ibrow_size[i]);
    }
    printf("\n");
    if(blr->symm == 'N')
    {
        printf("%d block columns (start, end):\n", blr->nbcols);
        i = 0;
        if(blr->nbcols > 0)
            printf("(%i, %i)", blr->ibcol_start[i],
                    blr->ibcol_start[i]+blr->ibcol_size[i]);
        for(i = 0; i < blr->nbcols; i++)
        {
            printf(", (%i, %i)", blr->ibcol_start[i],
                    blr->ibcol_start[i]+blr->ibcol_size[i]);
        }
        printf("\n");
    }
    printf("%d admissible blocks:\n", blr->admissible_nblocks);
    for(i = 0; i < blr->nbrows; i++)
    {
        j = blr->admissible_block_start[i];
        if(blr->admissible_block_size[i] > 0)
            printf("Admissible block columns for block row %d: %d", i,
                    blr->admissible_block[j]);
        for(k = 1; k < blr->admissible_block_size[i]; k++)
        {
            printf(" %d", blr->admissible_block[j+k]);
        }
        if(blr->admissible_block_size[i] > 0)
            printf("\n");
    }
    //printf("\n");
}

STARS_BLRmatrix *STARS_BLR_from_array(Array *array, int nbrows, int nbcols,
        int *ibrow_size, int *ibcol_size, double tol, int maxrank, int fixrank)
{
    if(array->ndim != 2)
    {
        fprintf(stderr, "Input array should be 2-dimensional\n");
        exit(1);
    }
    STARS_Problem *problem = STARS_Problem_from_array(array, 'N');
    STARS_BLR *format = (STARS_BLR *)malloc(sizeof(STARS_BLR));
    format->problem = problem;
    format->symm = problem->symm;
    format->nrows = array->shape[0];
    format->ncols = array->shape[1];
    format->nbrows = nbrows;
    format->nbcols = nbcols;
    format->ibrow_size = (int *)malloc(nbrows*sizeof(int));
    memcpy(format->ibrow_size, ibrow_size, nbrows);
    format->ibcol_size = (int *)malloc(nbcols*sizeof(int));
    memcpy(format->ibcol_size, ibcol_size, nbrows);
    format->ibrow_start = (int *)malloc(nbrows*sizeof(int));
    format->ibcol_start = (int *)malloc(nbcols*sizeof(int));
    format->ibrow_start[0] = 0;
    for(int i = 1; i < nbrows; i++)
        format->ibrow_start[i] = format->ibrow_start[i-1]+ibrow_size[i-1];
    format->ibcol_start[0] = 0;
    for(int i = 1; i < nbcols; i++)
        format->ibcol_start[i] = format->ibcol_start[i-1]+ibcol_size[i-1];
}


int batched_lowrank_approximation(STARS_BLRmatrix *mat, int count, int *id,
        int maxrank, double tol, void **UV, int *rank)
{
    int bi, i, j;
    STARS_BLR *format = mat->format;
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
                /*
                for(k = 0; k < trank; k++)
                    cblas_dcopy(cols, tV+k, rows, UV[bi]+sizeof(double)*
                            (rows*trank+k*cols), 1);
                for(k = 0; k < cols; k++)
                    cblas_dscal(trank, ptrS[k], UV[bi]+sizeof(double)*
                            (rows*trank+k), cols);
                */
                /*
                for(k = 0; k < cols; k++)
                {
                    cblas_dscal(trank, ptrS[k], tV+k*rows, 1);
                    cblas_dcopy(trank, tV+k*rows, 1, UV[bi]+sizeof(double)*
                            trank*(rows+k), 1);
                }
                */
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

int batched_get_block(STARS_BLRmatrix *mat, int count, int *id, void **A)
{
    STARS_BLR *format = mat->format;
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

STARS_BLRmatrix *STARS_blr_batched_algebraic_compress(STARS_BLR *format,
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
    STARS_BLRmatrix *mat = (STARS_BLRmatrix *)malloc(sizeof(STARS_BLRmatrix));
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

STARS_BLRmatrix *STARS_blr__compress_algebraic_svd(STARS_BLR *format,
        int maxrank, double tol, int KADIR)
    // Private function of STARS-H
    // Uses SVD to acquire rank of each block, compresses given matrix (given
    // by block kernel, which returns submatrices) with relative accuracy tol
    // or with given maximum rank (if maxrank <= 0, then tolerance is used)
{
    int i, j, k, l, bi, rows, cols, mn, rank, tmprank, error = 0, info;
    int shape[2];
    char symm = format->symm;
    //printf("'%c' format->symm\n", symm);
    Array *block, *block2;
    double *U, *S, *V, *ptr;
    double norm;
    STARS_BLRmatrix *mat = (STARS_BLRmatrix *)malloc(sizeof(STARS_BLRmatrix));
    mat->format = format;
    mat->bcount = format->nbrows * format->nbcols;
    mat->bindex = (int *)malloc(2*mat->bcount*sizeof(int));
    mat->brank = (int *)malloc(mat->bcount*sizeof(int));
    mat->U = (Array **)malloc(mat->bcount*sizeof(Array *));
    mat->V = (Array **)malloc(mat->bcount*sizeof(Array *));
    mat->A = (Array **)malloc(mat->bcount*sizeof(Array *));
    for(i = 0; i < format->nbrows; i++)
        for(j = 0; j < format->nbcols; j++)
            // Cycle over every block
        {
            bi = i * format->nbcols + j;
            mat->bindex[2*bi] = i;
            mat->bindex[2*bi+1] = j;
            rows = format->ibrow_size[i];
            cols = format->ibcol_size[j];
            mn = rows > cols ? cols : rows;
            if((i < j && symm == 'S') || error == 1)
            {
                mat->U[bi] = NULL;
                mat->V[bi] = NULL;
                mat->A[bi] = NULL;
                mat->brank[bi] = mn;
                continue;
            }
            shape[0] = rows;
            shape[1] = cols;
            block = Array_new(2, shape, format->problem->dtype, 'F');
            info = (format->problem->kernel)(rows, cols, format->row_pivot +
                    format->ibrow_start[i], format->col_pivot +
                    format->ibcol_start[j], format->problem->row_data,
                    format->problem->col_data, block->buffer);
            block2 = Array_copy(block, 'N');
            dmatrix_lr(rows, cols, block->buffer, tol, &rank, &U, &S, &V);
            //Array_SVD(block, &U, &S, &V);
            //rank = SVD_get_rank(S, tol, 'F');
            Array_free(block);
            if((KADIR == 0 && rank < mn/2) || (KADIR == 1 && i != j))
                // If block is low-rank
            {
                if(maxrank > 0)
                    // If block is low-rank and maximum rank is upperbounded,
                    // then rank should be equal to maxrank
                    rank = maxrank;
                mat->A[bi] = NULL;
                shape[0] = rows;
                shape[1] = rank;
                mat->U[bi] = Array_new(2, shape, 'd', 'F');
                shape[0] = rank;
                shape[1] = cols;
                mat->V[bi] = Array_new(2, shape, 'd', 'F');
                cblas_dcopy(rank*rows, U, 1, mat->U[bi]->buffer, 1);
                ptr = mat->V[bi]->buffer;
                for(k = 0; k < cols; k++)
                    for(l = 0; l < rank; l++)
                    {
                        ptr[k*rank+l] = S[l]*V[k*mn+l];
                    }
                mat->brank[bi] = rank;
                Array_free(block2);
            }
            else
                // If block is NOT low-rank
            {
                if(i != j && KADIR == 1)
                {
                    printf("FULL rank offdiagonal (%i,%i)\n", i, j);
                    error = 1;
                }
                mat->U[bi] = NULL;
                mat->V[bi] = NULL;
                mat->A[bi] = block2;
                mat->brank[bi] = mn;
            }
            free(U);
            free(S);
            free(V);
        }
    if(error == 1)
    {
        STARS_BLRmatrix_free(mat);
        return NULL;
    }
    return mat;
}

void STARS_BLRmatrix_info(STARS_BLRmatrix *mat)
    // Print information on each block of block low-rank matrix.
{
    int i, bi, bj, r;
    if(mat == NULL)
    {
        printf("STARS_BLRmatrix NOT initialized\n");
        return;
    }
    for(i = 0; i < mat->bcount; i++)
    {
        bi = mat->bindex[2*i];
        bj = mat->bindex[2*i+1];
        r = mat->brank[i];
        if(r != -1)
        {
            printf("block (%i, %i) U: ", bi, bj);
            Array_info(mat->U[i]);
            printf("block (%i, %i) V: ", bi, bj);
            Array_info(mat->V[i]);
        }
        else
        {
            printf("block (%i, %i): ", bi, bj);
            Array_info(mat->A[i]);
        }
    }
}

void STARS_BLRmatrix_free(STARS_BLRmatrix *mat)
    // Free memory, used by matrix
{
    int bi;
    char symm = mat->format->symm;
    if(mat == NULL)
    {
        printf("STARS_BLRmatrix NOT initialized\n");
        return;
    }
    for(bi = 0; bi < mat->bcount; bi++)
    {
        if(mat->A[bi] != NULL)
            Array_free(mat->A[bi]);
        if(mat->U[bi] != NULL)
            Array_free(mat->U[bi]);
        if(mat->V[bi] != NULL)
            Array_free(mat->V[bi]);
    }
    free(mat->A);
    free(mat->U);
    free(mat->V);
    free(mat->bindex);
    free(mat->brank);
    free(mat);
}



void STARS_BLRmatrix_error(STARS_BLRmatrix *mat)
{
    int bi, i, j;
    double diff = 0., norm = 0., tmpnorm, tmpdiff, tmperr, maxerr = 0.;
    int rows, cols, info, shape[2];
    STARS_BLR *format = mat->format;
    STARS_Problem *problem = format->problem;
    //Array *(*kernel)(int, int, int *, int *, void *, void *) =
    //    mat->format->problem->kernel;
    Array *block, *block2;
    char symm = format->symm;
    for(bi = 0; bi < mat->bcount; bi++)
    {
        i = mat->bindex[2*bi];
        j = mat->bindex[2*bi+1];
        if(i < j && symm == 'S')
            continue;
        rows = format->ibrow_size[i];
        cols = format->ibcol_size[j];
        shape[0] = rows;
        shape[1] = cols;
        block = Array_new(2, shape, problem->dtype, 'F');
        info = (problem->kernel)(rows, cols, format->row_pivot +
                format->ibrow_start[i], format->col_pivot +
                format->ibcol_start[j], problem->row_data,
                problem->col_data, block->buffer);
        tmpnorm = Array_norm(block);
        norm += tmpnorm*tmpnorm;
        if(i != j && symm == 'S')
            norm += tmpnorm*tmpnorm;
        if(mat->A[bi] == NULL)
        {
            block2 = Array_dot(mat->U[bi], mat->V[bi]);
            tmpdiff = Array_diff(block, block2);
            Array_free(block2);
            diff += tmpdiff*tmpdiff;
            if(i != j && symm == 'S')
                diff += tmpdiff*tmpdiff;
            tmperr = tmpdiff/tmpnorm;
            if(tmperr > maxerr)
                maxerr = tmperr;
        }
        /*
        if(mat->A[bi] != NULL)
        {
            block2 = mat->A[bi];
            tmpdiff = Array_diff(block, block2);
            diff += tmpdiff*tmpdiff;
        }
        */
        Array_free(block);
    }
    printf("Relative error of approximation of full matrix: %e\n",
            sqrt(diff/norm));
    printf("Maximum relative error of per-block approximation: %e\n", maxerr);
}

void STARS_BLRmatrix_getblock(STARS_BLRmatrix *mat, int i, int j, int pivot,
        int *shape, int *rank, void **U, void **V, void **A)
// PLEASE CLEAN MEMORY AFTER USE
{
    if(pivot != 'C' && pivot != 'F')
    {
        fprintf(stderr, "Parameter pivot should be 'C' or 'F', not '%c'\n",
                pivot);
        exit(1);
    }
    int bi = i * mat->format->nbcols + j;
    Array *tmp;
    *rank = mat->brank[bi];
    shape[0] = mat->format->ibrow_size[i];
    shape[1] = mat->format->ibcol_size[j];
    *U = NULL;
    *V = NULL;
    *A = NULL;
    if(mat->U[bi] != NULL)
    {
        tmp = Array_copy(mat->U[bi], pivot);
        *U = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
    if(mat->V[bi] != NULL)
    {
        tmp = Array_copy(mat->V[bi], pivot);
        *V = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
    if(mat->A[bi] != NULL)
    {
        tmp = Array_copy(mat->A[bi], pivot);
        *A = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
}

void STARS_BLR_getblock(STARS_BLR *format, int i, int j, int pivot, int *shape,
        void **A)
// PLEASE CLEAN MEMORY POINTER AFTER USE
{
    if(pivot != 'C' && pivot != 'F')
    {
        fprintf(stderr, "Parameter pivot should be 'C' or 'F', not '%c'\n",
                pivot);
        exit(1);
    }
    if(pivot == 'C')
    {
        fprintf(stderr, "pivot 'C' is not supported anymore\n");
        exit(1);
    }
    int rows = format->ibrow_size[i];
    int cols = format->ibcol_size[j];
    int info;
    shape[0] = rows;
    shape[1] = cols;
    *A = malloc(format->problem->dtype_size*rows*cols);
    info = (format->problem->kernel)(rows, cols, format->row_pivot +
            format->ibrow_start[i], format->col_pivot +
            format->ibcol_start[j], format->problem->row_data,
            format->problem->col_data, *A);
}

void STARS_BLRmatrix_printKADIR(STARS_BLRmatrix *mat)
{
    int i, j, bi;
    for(bi = 0; bi < mat->bcount; bi++)
    {
        i = mat->bindex[2*bi];
        j = mat->bindex[2*bi+1];
        if(i < j)
            continue;
        printf("BLOCK %d %d:\n", i, j);
        if(mat->A[bi] == NULL)
        {
            Array_info(mat->U[bi]);
            Array_print(mat->U[bi]);
            Array_info(mat->V[bi]);
            Array_print(mat->V[bi]);
        }
        else
        {
            Array_info(mat->A[bi]);
            Array_print(mat->A[bi]);
        }
        printf("\n");
    }
}

void STARS_BLRmatrix_heatmap(STARS_BLRmatrix *mat, char *fname)
{
    int i, j, bi;
    STARS_BLR *format = mat->format;
    FILE *fd = fopen(fname, "w");
    fprintf(fd, "%d %d\n", format->nbrows, format->nbcols);
    for(i = 0; i < format->nbrows; i++)
    {
        for(j = 0; j < format->nbrows; j++)
        {
            bi = i * format->nbcols + j;
            if(format->symm == 'S' && i < j)
                bi = j * format->nbcols + i;
            fprintf(fd, " %d", mat->brank[bi]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}
