#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include "misc.h"
#include "blr.h"
#include "cblas.h"


STARS_blrmat *STARS_blr_compress_uniform(int bcount, int bsize, void *data,
        block_func *kernel, double tol)
{
    // Compress matrix, divided uniformly into blocks
    // Parameters:
    //  bcount: number of block rows (block columns)
    //  bsize: number of rows (columns) of each block
    //  data: pointer to structure, holding all the information on the problem
    //  kernel: pointer to a block kernel, which returns submatrices
    //  tol: relative accuracy parameter
    // Returns:
    //  STARS_blrmat, matrix in a block low-rank format
    STARS_blrmat *A = (STARS_blrmat *)malloc(sizeof(STARS_blrmat));
    A->symm = 'N';
    A->dtype = 'd';
    A->rows = bsize*bcount;
    A->cols = A->rows;
    A->row_order = (int *)malloc(A->rows*sizeof(int));
    A->col_order = A->row_order;
    A->row_data = data;
    A->col_data = A->row_data;
    A->brows = bcount;
    A->bcols = A->brows;
    A->brow_start = (int *)malloc(A->brows*sizeof(int));
    A->bcol_start = A->brow_start;
    A->brow_size = (int *)malloc(A->brows*sizeof(int));
    A->bcol_size = A->brow_size;
    A->bcount = bcount*(bcount+1)/2;
    A->bindex = (int *)malloc(2*A->bcount*sizeof(int));
    A->brank = (int *)malloc(A->bcount*sizeof(int));
    A->U = malloc(A->bcount*sizeof(double *));
    A->V = malloc(A->bcount*sizeof(double *));
    A->A = malloc(A->bcount*sizeof(double *));
    A->kernel = kernel;
    int i, j, bi = 0;
    for(i = 0; i < A->rows; i++)
    {
        A->row_order[i] = i;
    }
    for(i = 0; i < bcount; i++)
    {
        A->brow_size[i] = bsize;
        A->brow_start[i] = bsize*i;
    }
    for(i = 0; i < bcount; i++)
    {
        for(j = 0; j <= i; j++)
        {
            A->bindex[2*bi] = i;
            A->bindex[2*bi+1] = j;
            bi += 1;
        }
    }
    STARS_blr__compress_algebraic_svd(A, tol);
    return A;
}

STARS_blrmat *STARS_blr_compress(int symm, int rows, int cols,
        void *row_data, void *col_data, int *row_order, int *col_order,
        int brows, int bcols, int *brow_start, int *bcol_start, int *brow_size,
        int *bcol_size, block_func kernel, double tol)
{
    // Compress matrix, divided into blocks (no hierarchy, tensor grid of
    // blocks)
    // Parameters:
    //  symm: 'S' if matrix is symmetric, 'N' otherwise
    //  rows: total number of rows of a matrix
    //  cols: total number of columns of a matrix
    //  row_data: pointer to structure, corresponding to rows
    //  col_data: pointer to structure, corresponding to columns
    //  row_order: permutation of rows
    //  col_order: permutation of columns
    //  brows: number of block rows
    //  bcols: number of block columns
    //  brow_start: array of start points in row_order of each block row
    //  bcol_start: start points in col_order of each block column
    //  brow_size: array of number of rows of each block row
    //  bcol_size: array of number of columns of each block column
    //  kernel: pointer to a block kernel, which generates submatrices on given
    //      rows and columns
    //  tol: relative accuracy parameter
    // Returns:
    //  STARS_blrmat, matrix in a block low-rank format
    int i, j, bi = 0, ii;
    STARS_blrmat *A = (STARS_blrmat *)malloc(sizeof(STARS_blrmat));
    A->symm = symm;
    A->dtype = 'd';
    A->rows = rows;
    A->row_order = row_order;
    A->row_data = row_data;
    A->brows = brows;
    A->brow_start = brow_start;
    A->brow_size = brow_size;
    if(symm == 'S')
    {
        A->cols = rows;
        A->col_order = row_order;
        A->col_data = row_data;
        A->bcols = brows;
        A->bcol_start = brow_start;
        A->bcol_size = brow_size;
        A->bcount = brows*(brows+1)/2;
        A->bindex = (int *)malloc(2*A->bcount*sizeof(int));
        for(i = 0; i < brows; i++)
        {
            for(j = 0; j <= i; j++)
            {
                A->bindex[2*bi] = i;
                A->bindex[2*bi+1] = j;
                bi += 1;
            }
        }
    }
    else if(symm == 'N')
    {
        A->cols = cols;
        A->col_order = col_order;
        A->col_data = col_data;
        A->bcols = bcols;
        A->bcol_start = bcol_start;
        A->bcol_size = bcol_size;
        A->bcount = brows*bcols;
        A->bindex = (int *)malloc(2*A->bcount*sizeof(int));
        for(i = 0; i < brows; i++)
        {
            ii = 2*i*bcols;
            for(j = 0; j < bcols; j++)
            {
                A->bindex[ii+2*j] = i;
                A->bindex[ii+2*j+1] = j;
            }
        }
    }
    else
    {
        printf("Wrong parameter 1 in compress_blr\n");
        free(A);
        return NULL;
    }
    A->bindex = (int *)malloc(A->bcount*2*sizeof(int));
    A->brank = (int *)malloc(A->bcount*sizeof(int));
    A->U = malloc(A->bcount*sizeof(double *));
    A->V = malloc(A->bcount*sizeof(double *));
    A->A = malloc(A->bcount*sizeof(double *));
    A->kernel = kernel;
    STARS_blr__compress_algebraic_svd(A, tol);
    return A;
}


void STARS_blr__compress_algebraic_svd(STARS_blrmat *A, double tol)
{
    // Private function of STARS-H
    // Uses SVD to acquire rank of each block, compresses given matrix (given
    // by block kernel, which retruns submatrices) with relative accuracy tol
    int i, j, k, l, bi, rows, cols, mn, rank;
    double *U, *S, *V, *ptr, *block, *block2;
    double norm;
    for(bi = 0; bi < A->bcount; bi++)
    {
        i = A->bindex[2*bi];
        j = A->bindex[2*bi+1];
        rows = A->brow_size[i];
        cols = A->bcol_size[j];
        block = (double *)malloc(rows*cols*sizeof(double));
        block2 = (double *)malloc(rows*cols*sizeof(double));
        mn = rows > cols ? cols : rows;
        A->kernel(rows, cols, A->row_order+A->brow_start[i],
                A->col_order+A->bcol_start[j],
                A->row_data, A->col_data, block);
        norm = cblas_dnrm2(rows*cols, block, 1);
        cblas_dcopy(rows*cols, block, 1, block2, 1);
        dmatrix_lr(rows, cols, block, tol, &rank, &U, &S, &V);
        free(block);
        A->brank[bi] = rank;
        if(rank < mn/2)
        {
            A->A[bi] = NULL;
            A->U[bi] = (double *)malloc(rows*rank*sizeof(double));
            A->V[bi] = (double *)malloc(rank*cols*sizeof(double));
            cblas_dcopy(rank*rows, U, 1, A->U[bi], 1);
            ptr = A->V[bi];
            for(k = 0; k < cols; k++)
                for(l = 0; l < rank; l++)
                {
                    ptr[k*rank+l] = S[l]*V[k*mn+l];
                }
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows, cols,
                    rank, 1., A->U[bi], rows, A->V[bi], rank, -1.,
                    block2, rows);
            printf("Low rank: %i %i %f %f\n", i, j, cblas_dnrm2(rows*cols,
                    block2, 1)/norm, norm);
            free(block2);
        }
        else
        {
            A->U[bi] = NULL;
            A->V[bi] = NULL;
            A->A[bi] = block2;
            printf("Full rank: %i %i %f\n", i, j, norm);
        }
        free(U);
        free(S);
        free(V);
    }
}
