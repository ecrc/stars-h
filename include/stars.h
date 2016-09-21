#ifndef _STARS_H_
#define _STARS_H_

typedef void *(*block_kernel)(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result);

typedef struct STARS_Problem
{
    int nrows, ncols;
    char symm, dtype;
    void *row_data, *col_data;
    block_kernel kernel;
} STARS_Problem;

typedef struct STARS_BLR
{
    STARS_Problem *problem;
    char symm;
    int nrows, ncols;
    int *row_order, *col_order;
    int nbrows, nbcols;
    int *ibrow_start, *ibcol_start;
    int *ibrow_size, *ibcol_size;
} STARS_BLR;

typedef struct STARS_BLRmatrix
{
    STARS_Problem *problem;
    STARS_BLR *format;
    int bcount, *bindex, *brank;
    double **U, **V, **A;
} STARS_BLRmatrix;

STARS_BLRmatrix *STARS_blr__compress_algebraic_svd(STARS_BLR *format,
        double tol);
#endif // _STARS_H_
