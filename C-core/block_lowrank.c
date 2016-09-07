#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include "misc.h"
#include "block_lowrank.h"
#include "cblas.h"


STARS_blr *unified_compress_symm(int block_count, int block_size, void *data,
        block_func kernel, double tol)
{
    STARS_blr *A = (STARS_blr *)malloc(sizeof(STARS_blr));
    A->symm = 'N';
    A->dtype = 'd';
    A->rows = block_size*block_count;
    A->cols = A->rows;
    A->row_order = (int *)malloc(A->rows*sizeof(int));
    A->col_order = A->row_order;
    A->row_data = data;
    A->col_data = A->row_data;
    A->row_bcount = block_count;
    A->col_bcount = A->row_bcount;
    A->row_bstart = (int *)malloc(A->row_bcount*sizeof(int));
    A->col_bstart = A->row_bstart;
    A->row_bsize = (int *)malloc(A->row_bcount*sizeof(int));
    A->col_bsize = A->row_bsize;
    A->block_count = block_count*(block_count+1)/2;
    A->block_index = (int *)malloc(A->block_count*2*sizeof(int));
    A->block_rank = (int *)malloc(A->block_count*sizeof(int));
    A->U = malloc(A->block_count*sizeof(double *));
    A->V = malloc(A->block_count*sizeof(double *));
    A->A = malloc(A->block_count*sizeof(double *));
    A->kernel = kernel;
    int i, j, bi = 0;
    for(i = 0; i < A->rows; i++)
    {
        A->row_order[i] = i;
    }
    for(i = 0; i < A->row_bcount; i++)
    {
        A->row_bsize[i] = block_size;
        A->row_bstart[i] = block_size*i;
    }
    for(i = 0; i < block_count; i++)
    {
        for(j = 0; j < i; j++)
        {
            A->block_index[2*bi] = i;
            A->block_index[2*bi+1] = j;
            bi += 1;
        }
    }
    for(i = 0; i < block_count; i++)
    {
        A->block_index[2*bi] = i;
        A->block_index[2*bi+1] = i;
        bi += 1;
    }
    compress_blr(A, tol);
    return A;
}

void compress_blr(STARS_blr *A, double tol)
{
    int i, j, k, l, bi, rows, cols, mn, rank;
    double *U, *S, *V, *ptr, *block, *block2;
    double norm;
    for(bi = 0; bi < A->block_count; bi++)
    {
        i = A->block_index[2*bi];
        j = A->block_index[2*bi+1];
        rows = A->row_bsize[i];
        cols = A->col_bsize[j];
        block = (double *)malloc(rows*cols*sizeof(double));
        block2 = (double *)malloc(rows*cols*sizeof(double));
        mn = rows > cols ? cols : rows;
        //dmatrix_print(1, rows, A->row_order+A->row_bstart[i]);
        //dmatrix_print(1, cols, A->col_order+A->col_bstart[j]);
        A->kernel(rows, cols, A->row_order+A->row_bstart[i],
                A->col_order+A->col_bstart[j],
                A->row_data, A->col_data, block);
        norm = cblas_dnrm2(rows*cols, block, 1);
        cblas_dcopy(rows*cols, block, 1, block2, 1);
        dmatrix_lr(rows, cols, block, tol, &rank, &U, &S, &V);
        free(block);
        A->block_rank[bi] = rank;
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

/*
dblr_matrix *compress(int block_size, int block_count, double *points,
        void (*block_kernel)(int, double *, int, int, int, double (*)(double),
        double *), double (*kernel)(double), double tol)
{
    dblr_matrix *A = (dblr_matrix *)malloc(sizeof(dblr_matrix));
    A->U = NULL;
    A->V = NULL;
    int nblocks = block_count*block_count;
    A->offset_U = (int *)malloc(nblocks*sizeof(int));
    A->offset_V = (int *)malloc(nblocks*sizeof(int));
    int i, j, k, l;
    double *block = (double *)malloc(block_size*block_size*sizeof(double));
    double *tmp_block = (double *)malloc(block_size*block_size*sizeof(double));
    int rank;
    double *U, *S, *V;
    double *ptr;
    double norm;
    // Cycle bu block rows
    for(i = 0; i < block_count; i++)
    {
        // Cycle by block columns
        for(j = 0; j < block_count; j++)
        {
            block_kernel(block_count*block_size, points, i*block_size,
                    j*block_size, block_size, kernel, block);
            norm = cblas_dnrm2(block_size*block_size, block, 1);
            cblas_dcopy(block_size*block_size, block, 1, tmp_block, 1);
            dmatrix_lr(block_size, block_size, block, tol, &rank, &U, &S, &V);
            A->U = (double *)realloc(A->U, (A->size_U+rank*block_size)
                    *sizeof(double));
            ptr = A->U+A->size_U;
            for(k = 0; k < rank*block_size; k++)
            {
                ptr[k] = U[k];
            }
            A->offset_U[i*block_count+j] = A->size_U;
            A->size_U += rank*block_size;
            A->V = (double *)realloc(A->V, (A->size_V+rank*block_size)
                    *sizeof(double));
            ptr = A->V+A->size_V;
            for(k = 0; k < block_size; k++)
                for(l = 0; l < rank; l++)
                {
                    ptr[k*rank+l] = S[l]*V[k*block_size+l];
                }
            A->offset_V[i*block_count+j] = A->size_V;
            A->size_V += rank*block_size;
            for(k = 0; k < block_size; k++)
                for(l = 0; l < block_size; l++)
                {
                    U[k*block_size+l] = S[k]*U[k*block_size+l];
                }
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, block_size,
                    block_size, rank, 1., A->U+A->offset_U[i*block_count+j],
                    block_size, A->V+A->offset_V[i*block_size+j], rank,
                    -1., tmp_block, block_size);
            printf("%i %i %f %f\n", i, j, cblas_dnrm2(block_size*block_size,
                        tmp_block, 1)/norm, norm);
            free(U);
            free(S);
            free(V);
        }
    }
    free(block);
    free(tmp_block);
    return A;
}
*/
