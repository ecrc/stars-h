#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include "stars.h"
#include "stars-misc.h"
#include "cblas.h"
#include "lapacke.h"


int batched_lowrank_approximation(STARS_BLRmatrix *mat, int count, int *id,
        void **UV, int maxrank)
{
    int bi, i, j;
    STARS_BLR *format = mat->format;
    int (*kernel)(int, int, int *, int *, void *, void *, void *) =
        format->problem->kernel;
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
    char *block, *work, *iwork, *U, *S, *V;
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
    #pragma omp parallel shared(block, work, iwork, U, V, S) private(i, j, bi)
    {
        int nthreads;
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
            printf("Total threads %d\n", nthreads);
            block = malloc(nthreads*block_size);
            work = malloc(nthreads*lwork);
            iwork = malloc(nthreads*liwork);
            U = malloc(nthreads*block_size);
            V = malloc(nthreads*block_size);
            S = malloc(nthreads*mx*S_dtype_size);
            //printf("block_size %d, S_size %d\n", block_size, mx*S_dtype_size);
        }
        #pragma omp barrier
        int tid = omp_get_thread_num();
        char *tblock = block+block_size*tid;
        char *twork = work+lwork*tid;
        char *tiwork = iwork+liwork*tid;
        char *tU = U+block_size*tid;
        char *tV = V+block_size*tid;
        char *tS = S+mx*S_dtype_size*tid;
        int tinfo = 0;
        //printf("Work in thread %d\n", tid);
        #pragma omp for
        for(bi = 0; bi < count; bi++)
        {
            i = mat->bindex[2*bi];
            j = mat->bindex[2*bi+1];
            kernel(format->ibrow_size[i], format->ibcol_size[j],
                    format->row_order +
                    format->ibrow_start[i], format->col_order +
                    format->ibcol_start[j], format->problem->row_data,
                    format->problem->col_data, tblock);
            //printf("%d %d\n", format->ibrow_size[i], format->ibcol_size[j]);
            tinfo = LAPACKE_dgesdd_work(LAPACK_COL_MAJOR, 'S',
                    format->ibrow_size[i],
                    format->ibcol_size[j], tblock, format->ibrow_size[i], tS,
                    tU, format->ibrow_size[i], tV, format->ibcol_size[j],
                    twork, tlwork, tiwork);
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
        #pragma omp barrier
    }
    return 0;
}

int batched_get_block(STARS_BLR *format, int count, int *block_id, void **A)
{
    return 0;
}

STARS_BLRmatrix *STARS_blr_batched_algebraic_compress(STARS_BLR *format,
        int maxrank, double tol)
{
    int i, j, bi, mn;
    char symm = format->symm;
    int num_blocks = format->nbrows*format->nbcols;
    if(symm == 'S')
        num_blocks = (format->nbrows+1)*format->nbrows/2;
    int *batched_block_id = (int *)malloc(num_blocks*sizeof(int));
    int batched_id = 0;
    STARS_BLRmatrix *mat = (STARS_BLRmatrix *)malloc(sizeof(STARS_BLRmatrix));
    mat->format = format;
    mat->bindex = (int *)malloc(2*format->nbcols*format->nbrows*sizeof(int));
    for(i = 0; i < format->nbrows; i++)
        for(j = 0; j < format->nbcols; j++)
        {
            bi = i * format->nbcols + j;
            mat->bindex[2*bi] = i;
            mat->bindex[2*bi+1] = j;
            if(i < j && symm == 'S')
            {
                //mat->U[bi] = NULL;
                //mat->V[bi] = NULL;
                //mat->A[bi] = NULL;
                //mat->brank[bi] = -1;
                continue;
            }
            //printf("bid %d\n", batched_id);
            batched_block_id[batched_id] = bi;
            batched_id++;
        }
    int rows = format->ibrow_size[0], cols = format->ibcol_size[0];
    int uv_size = (rows+cols)*maxrank*sizeof(double);
    void *UV_array = malloc(uv_size*num_blocks);
    void **UV = malloc(num_blocks*sizeof(void *));
    for(i = 0; i < num_blocks; i++)
        UV[i] = UV_array+uv_size*i;
    batched_lowrank_approximation(mat, num_blocks, batched_block_id, UV, maxrank);
    return mat;
}

STARS_BLRmatrix *STARS_blr__compress_algebraic_svd(STARS_BLR *format,
        int maxrank, double tol, int KADIR)
    // Private function of STARS-H
    // Uses SVD to acquire rank of each block, compresses given matrix (given
    // by block kernel, which returns submatrices) with relative accuracy tol
    // or with given maximum rank (if maxrank <= 0, then tolerance is used)
{
    int i, j, k, l, bi, rows, cols, mn, rank, tmprank, error = 0;
    int shape[2];
    char symm = format->symm;
    //printf("'%c' format->symm\n", symm);
    Array *block, *block2;
    double *U, *S, *V, *ptr;
    double norm;
    STARS_BLRmatrix *mat = (STARS_BLRmatrix *)malloc(sizeof(STARS_BLRmatrix));
    mat->format = format;
    mat->problem = format->problem;
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
            block = (mat->problem->kernel)(rows, cols, format->row_order +
                    format->ibrow_start[i], format->col_order +
                    format->ibcol_start[j], format->problem->row_data,
                    format->problem->col_data);
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

void STARS_BLR_info(STARS_BLR *format)
    // Print onfo on block partitioning
{
    int i;
    if(format == NULL)
    {
        printf("STARS_BLR NOT initialized\n");
        return;
    }
    if(format->symm == 'S')
        printf("Symmetric partitioning into blocks\n(blocking for columns "
                "is the same, as blocking for rows)\n");
    printf("Block rows (start, end):");
    i = 0;
    if(format->nbrows > 0)
        printf(" (%i, %i)", format->ibrow_start[i],
                format->ibrow_start[i]+format->ibrow_size[i]);
    for(i = 1; i < format->nbrows; i++)
    {
        printf(", (%i, %i)", format->ibrow_start[i],
                format->ibrow_start[i]+format->ibrow_size[i]);
    }
    printf("\n");
    if(format->symm == 'N')
    {
        printf("Block columns (start, end):");
        i = 0;
        if(format->nbcols > 0)
            printf(" (%i, %i)", format->ibcol_start[i],
                    format->ibcol_start[i]+format->ibcol_size[i]);
        for(i = 0; i < format->nbcols; i++)
        {
            printf("(%i, %i), ", format->ibcol_start[i],
                    format->ibcol_start[i]+format->ibcol_size[i]);
        }
        printf("\n");
    }
}

void STARS_BLR_free(STARS_BLR *format)
    // Free memory, used by block partitioning data
{
    if(format == NULL)
    {
        printf("STARS_BLR NOT initialized\n");
        return;
    }
    free(format->row_order);
    free(format->ibrow_start);
    free(format->ibrow_size);
    if(format->symm == 'N')
    {
       free(format->col_order);
       free(format->ibcol_start);
       free(format->ibcol_size);
    }
    free(format);
}

void STARS_BLRmatrix_error(STARS_BLRmatrix *mat)
{
    int bi, i, j;
    double diff = 0., norm = 0., tmpnorm, tmpdiff, tmperr, maxerr = 0.;
    int rows, cols;
    STARS_BLR *format = mat->format;
    Array *block, *block2;
    char symm = mat->format->symm;
    for(bi = 0; bi < mat->bcount; bi++)
    {
        i = mat->bindex[2*bi];
        j = mat->bindex[2*bi+1];
        if(i < j && symm == 'S')
            continue;
        rows = format->ibrow_size[i];
        cols = format->ibcol_size[j];
        block = (mat->problem->kernel)(rows, cols, format->row_order +
                format->ibrow_start[i], format->col_order +
                format->ibcol_start[j], format->problem->row_data,
                format->problem->col_data);
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
        Array_free(block);
    }
    printf("Relative error of approximation of full matrix: %e\n",
            sqrt(diff/norm));
    printf("Maximum relative error of per-block approximation: %e\n", maxerr);
}

void STARS_BLRmatrix_getblock(STARS_BLRmatrix *mat, int i, int j, int order,
        int *shape, int *rank, void **U, void **V, void **A)
// PLEASE CLEAN MEMORY AFTER USE
{
    if(order != 'C' && order != 'F')
    {
        fprintf(stderr, "Parameter order should be 'C' or 'F', not '%c'\n",
                order);
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
        tmp = Array_copy(mat->U[bi], order);
        *U = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
    if(mat->V[bi] != NULL)
    {
        tmp = Array_copy(mat->V[bi], order);
        *V = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
    if(mat->A[bi] != NULL)
    {
        tmp = Array_copy(mat->A[bi], order);
        *A = tmp->buffer;
        free(tmp->shape);
        free(tmp->stride);
        free(tmp);
    }
}

void STARS_BLR_getblock(STARS_BLR *format, int i, int j, int order, int *shape,
        void **A)
// PLEASE CLEAN MEMORY POINTER AFTER USE
{
    if(order != 'C' && order != 'F')
    {
        fprintf(stderr, "Parameter order should be 'C' or 'F', not '%c'\n",
                order);
        exit(1);
    }
    int rows = format->ibrow_size[i];
    int cols = format->ibcol_size[j];
    Array *tmp, *tmp2;
    shape[0] = rows;
    shape[1] = cols;
    tmp = (format->problem->kernel)(rows, cols, format->row_order +
            format->ibrow_start[i], format->col_order +
            format->ibcol_start[j], format->problem->row_data,
            format->problem->col_data);
    tmp2 = Array_copy(tmp, order);
    *A = tmp2->buffer;
    Array_free(tmp);
    free(tmp2->shape);
    free(tmp2->stride);
    free(tmp2);
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
