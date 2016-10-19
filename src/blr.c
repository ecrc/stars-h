#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "stars.h"
#include "stars-misc.h"
#include "cblas.h"

STARS_BLRmatrix *STARS_blr__compress_algebraic_svd(STARS_BLR *format,
        int maxrank, double tol, int KADIR)
    // Private function of STARS-H
    // Uses SVD to acquire rank of each block, compresses given matrix (given
    // by block kernel, which returns submatrices) with relative accuracy tol
    // or with given maximum rank (if maxrank <= 0, then tolerance is used)
{
    int i, j, k, l, bi, rows, cols, mn, rank, tmprank, error = 0;
    double *U, *S, *V, *ptr;
    int shape[2];
    char symm = format->symm;
    //printf("'%c' format->symm\n", symm);
    Array *block, *block2;
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
            if((i < j && symm == 'S') || error == 1)
            {
                mat->U[bi] = NULL;
                mat->V[bi] = NULL;
                mat->A[bi] = NULL;
                mat->brank[bi] = 0;
                continue;
            }
            rows = format->ibrow_size[i];
            cols = format->ibcol_size[j];
            mn = rows > cols ? cols : rows;
            //printf("1");
            /*block = (mat->problem->kernel)(rows, cols, format->row_order +
                    format->ibrow_start[i], format->col_order +
                    format->ibcol_start[j], format->problem->row_data,
                    format->problem->col_data);*/
            shape[0] = rows;
            shape[1] = cols;
            printf("%d(%d) %d(%d) %d\n", format->ibrow_start[i], rows, format->ibcol_start[j], cols, mn);
            block = Array_new(2, shape, 'd', 'F');
            Array_init_ones(block);
            Array_print(block);
            //norm = cblas_dnrm2(rows*cols, block->buffer, 1);
            //printf("2");
            block2 = Array_copy(block);
            //Array_info(block);
            //printf("%d %d\n", rows, cols);
            dmatrix_lr(rows, cols, block->buffer, tol, &rank, &U, &S, &V);
            Array_free(block);
            //printf("3\n");
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
        printf("CLEARING\n");
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
            tmpdiff = Array_error(block, block2);
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

void STARS_BLRmatrix_getblock(STARS_BLRmatrix *mat, int i, int j,
        int *block_size, int *rank, void **U, void **V, void **A)
// DO NOT FREE POINTERS OF THIS FUNCTIONS
{
    int bi = i * mat->format->nbcols + j;
    *rank = mat->brank[bi];;
    *block_size = mat->format->ibrow_size[bi];
    *U = mat->U[bi]->buffer;
    *V = mat->V[bi]->buffer;
    *A = mat->A[bi]->buffer;
}

void STARS_BLR_getblock(STARS_BLR *format, int i, int j, int *block_size,
        void **A)
// PLEASE CLEAN MEMORY POINTER AFTER USE
{
    int rows = format->ibrow_size[i];
    int cols = format->ibcol_size[j];
    *block_size = rows;
    *A = (format->problem->kernel)(rows, cols, format->row_order +
            format->ibrow_start[i], format->col_order +
            format->ibcol_start[j], format->problem->row_data,
            format->problem->col_data);
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
