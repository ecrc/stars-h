#include <stdio.h>
#include <stdlib.h>
#include "stars.h"
#include "stars-misc.h"
#include "cblas.h"

STARS_BLRmatrix *STARS_blr__compress_algebraic_svd(STARS_BLR *format,
        double tol)
    // Private function of STARS-H
    // Uses SVD to acquire rank of each block, compresses given matrix (given
    // by block kernel, which returns submatrices) with relative accuracy tol
{
    int i, j, k, l, bi, rows, cols, mn, rank;
    double *U, *S, *V, *ptr, *block, *block2;
    double norm;
    STARS_BLRmatrix *mat = (STARS_BLRmatrix *)malloc(sizeof(STARS_BLRmatrix));
    mat->format = format;
    mat->problem = format->problem;
    mat->bcount = format->nbrows * format->nbcols;
    mat->bindex = (int *)malloc(2*mat->bcount*sizeof(int));
    mat->brank = (int *)malloc(mat->bcount*sizeof(int));
    mat->U = (double **)malloc(mat->bcount*sizeof(double *));
    mat->V = (double **)malloc(mat->bcount*sizeof(double *));
    mat->A = (double **)malloc(mat->bcount*sizeof(double *));
    for(i = 0; i < format->nbrows; i++)
        for(j = 0; j < format->nbcols; j++)
            // Cycle over every block
        {
            bi = i * format->nbcols + j;
            mat->bindex[2*bi] = i;
            mat->bindex[2*bi+1] = j;
            rows = format->ibrow_size[i];
            cols = format->ibcol_size[j];
            block = (double *)malloc(rows*cols*sizeof(double));
            block2 = (double *)malloc(rows*cols*sizeof(double));
            mn = rows > cols ? cols : rows;
            (mat->problem->kernel)(rows, cols, format->row_order +
                    format->ibrow_start[i], format->col_order +
                    format->ibcol_start[j], format->problem->row_data,
                    format->problem->col_data, block);
            norm = cblas_dnrm2(rows*cols, block, 1);
            cblas_dcopy(rows*cols, block, 1, block2, 1);
            dmatrix_lr(rows, cols, block, tol, &rank, &U, &S, &V);
            free(block);
            if(rank < mn/2)
                // If block is low-rank
            {
                mat->A[bi] = NULL;
                mat->U[bi] = (double *)malloc(rows*rank*sizeof(double));
                mat->V[bi] = (double *)malloc(rank*cols*sizeof(double));
                cblas_dcopy(rank*rows, U, 1, mat->U[bi], 1);
                ptr = mat->V[bi];
                for(k = 0; k < cols; k++)
                    for(l = 0; l < rank; l++)
                    {
                        ptr[k*rank+l] = S[l]*V[k*mn+l];
                    }
                mat->brank[bi] = rank;
                free(block2);
            }
            else
                // If block is NOT low-rank
            {
                mat->U[bi] = NULL;
                mat->V[bi] = NULL;
                mat->A[bi] = block2;
                mat->brank[bi] = -1;
            }
            free(U);
            free(S);
            free(V);
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
            printf("block (%i, %i): %i x %i matrix of rank %i\n", bi, bj,
                    mat->format->ibrow_size[bi], mat->format->ibcol_size[bj],
                    r);
        }
        else
        {
            printf("block (%i, %i): %i x %i full-rank matrix\n", bi, bj,
                    mat->format->ibrow_size[bi], mat->format->ibcol_size[bj]);
        }
    }
}

void STARS_BLRmatrix_free(STARS_BLRmatrix *mat)
    // Free memory, used by matrix
{
    int bi;
    if(mat == NULL)
    {
        printf("STARS_BLRmatrix NOT initialized\n");
        return;
    }
    for(bi = 0; bi < mat->bcount; bi++)
    {
        if(mat->A[bi] != NULL)  free(mat->A[bi]);
        if(mat->U[bi] != NULL)  free(mat->U[bi]);
        if(mat->V[bi] != NULL)  free(mat->V[bi]);
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
