#include <stdio.h>
#include <stdlib.h>
#include "stars.h"
#include "stars-misc.h"
#include "cblas.h"

STARS_BLRmatrix *STARS_blr__compress_algebraic_svd(STARS_BLR *format,
        double tol)
{
    // Private function of STARS-H
    // Uses SVD to acquire rank of each block, compresses given matrix (given
    // by block kernel, which retruns submatrices) with relative accuracy tol
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
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows,
                        cols, rank, 1., mat->U[bi], rows, mat->V[bi], rank,
                        -1., block2, rows);
                printf("Low rank: %i %i %f %f\n", i, j, cblas_dnrm2(rows*cols,
                        block2, 1)/norm, norm);
                mat->brank[bi] = rank;
                free(block2);
            }
            else
            {
                mat->U[bi] = NULL;
                mat->V[bi] = NULL;
                mat->A[bi] = block2;
                mat->brank[bi] = -1;
                printf("Full rank: %i %i %f\n", i, j, norm);
            }
            free(U);
            free(S);
            free(V);
        }
}
