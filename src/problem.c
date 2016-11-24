#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <complex.h>
#include "stars.h"
#include "misc.h"

int STARS_Problem_new(STARS_Problem **P, int ndim, int *shape, char symm,
        char dtype, void *row_data, void *col_data, block_kernel kernel,
        char *name)
// Init for STARS_Problem instance
// Parameters:
//   ndim: dimensionality of corresponding array. Equal 2+dimensionality of
//     kernel
//   shape: shape of corresponding array. shape[1:ndim-2] is equal to shape of
//     kernel
//   symm: 'S' for summetric P, 'N' for nonsymmetric P. Symmetric
//     P require symmetric kernel and equality of row_data and col_data
//   dtype: data type of the P. Equal to 's', 'd', 'c' or 'z' as in
//     LAPACK routines. Stands for data type of an element of a kernel.
//   row_data: pointer to some structure of physical data for rows
//   col_data: pointer to some srructure of physical data for columns
//   kernel: pointer to a function of interaction. More on this is written
//     somewhere else.
//   name: string, containgin name of the P. Used only to print
//     information about structure P.
//  Returns:
//    STARS_Problem *: pointer to structure P with proper filling of all
//    the fields of structure.
{
    if(P == NULL)
    {
        STARS_error("STARS_Problem_new", "invalid value of `P`");
        return 1;
    }
    if(ndim < 2)
    {
        STARS_error("STARS_Problem_new", "invalid value of `ndim`");
        return 1;
    }
    if(shape == NULL)
    {
        STARS_error("STARS_Problem_new", "invalid value of `shape`");
        return 1;
    }
    int i;
    size_t dtype_size = 0;
    if(dtype == 's')
        dtype_size = sizeof(float);
    else if(dtype == 'd')
        dtype_size = sizeof(double);
    else if(dtype == 'c')
        dtype_size = sizeof(float complex);
    else if(dtype == 'z')
        dtype_size = sizeof(double complex);
    else
    {
        STARS_error("STARS_Problem_new", "invalid value of `dtype`");
        return 1;
    }
    size_t entry_size = dtype_size;
    for(i = 1; i < ndim-1; i++)
        entry_size *= shape[i];
    *P = malloc(sizeof(**P));
    STARS_Problem *P2 = *P;
    if(P2 == NULL)
    {
        STARS_error("STARS_Problem_new", "malloc() failed");
        return 1;
    }
    P2->ndim = ndim;
    P2->shape = malloc(ndim*sizeof(*P2->shape));
    if(P2->shape == NULL)
    {
        STARS_error("STARS_Problem_new", "malloc() failed");
        return 1;
    }
    memcpy(P2->shape, shape, ndim*sizeof(*P2->shape));
    P2->symm = symm;
    P2->dtype = dtype;
    P2->dtype_size = dtype_size;
    P2->entry_size = entry_size;
    P2->row_data = row_data;
    P2->col_data = col_data;
    P2->kernel = kernel;
    P2->name = NULL;
    if(name != NULL)
    {
        P2->name = malloc(strlen(name)+1);
        if(P2->name == NULL)
        {
            STARS_error("STARS_Problem_new", "malloc() failed");
            return 1;
        }
        strcpy(P2->name, name);
    }
    return 0;
}

int STARS_Problem_free(STARS_Problem *P)
// Free memory, consumed by data buffers of data
{
    if(P == NULL)
    {
        STARS_error("STARS_Problem_free", "invalid value of `P`");
        return 1;
    }
    free(P->shape);
    if(P->name != NULL)
        free(P->name);
    free(P);
    return 0;
}

int STARS_Problem_info(STARS_Problem *P)
// Print some info about Problem
{
    if(P == NULL)
    {
        STARS_error("STARS_Problem_info", "invalid value of `P`");
        return 1;
    }
    printf("<STARS_Problem at %p, name \"%s\", shape (%d", P, P->name,
            P->shape[0]);
    for(int i = 1; i < P->ndim; i++)
        printf(",%d", P->shape[i]);
    printf("), '%c' dtype, '%c' symmetric>\n", P->dtype, P->symm);
    return 0;
}

int STARS_Problem_get_block(STARS_Problem *P, int nrows, int ncols, int *irow,
        int *icol, Array **A)
// Get submatrix on given rows and columns (rows=first dimension, columns=last
// dimension)
{
    if(P == NULL)
    {
        STARS_error("STARS_Problem_get_block", "invalid value of `P`");
        return 1;
    }
    if(irow == NULL)
    {
        STARS_error("STARS_Problem_get_block", "invalid value of `irow`");
        return 1;
    }
    if(icol == NULL)
    {
        STARS_error("STARS_Problem_get_block", "invalid value of `icol`");
        return 1;
    }
    int ndim = P->ndim, info;
    if(nrows < 0)
    {
        STARS_error("STARS_Problem_get_block", "invalid value of `nrows`");
        return 1;
    }
    if(ncols < 0)
    {
        STARS_error("STARS_Problem_get_block", "invalid value of `ncols`");
        return 1;
    }
    int *shape = malloc(ndim*sizeof(*shape));
    if(shape == NULL)
    {
        STARS_error("STARS_Problem_get_block", "malloc() failed");
        return 1;
    }
    shape[0] = nrows;
    shape[ndim-1] = ncols;
    memcpy(shape+1, P->shape+1, (ndim-2)*sizeof(*shape));
    info = Array_new(A, ndim, shape, P->dtype, 'F');
    if(info != 0)
        return info;
    info = P->kernel(nrows, ncols, irow, icol, P->row_data, P->col_data,
            (*A)->data);
    return info;
}

static int _matrix_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result)
{
    Array *A = row_data;
    size_t esize = A->dtype_size;
    size_t i, j, dest, src, lda;
    for(i = 1; i < A->ndim-1; i++)
        esize *= A->shape[i];
    if(A->order == 'C')
    {
        lda = A->shape[A->ndim-1];
        #pragma omp parallel for private(dest, src, i, j)
        for(i = 0; i < nrows; i++)
            for(j = 0; j < ncols; j++)
            {
                dest = j*nrows+i;
                src = irow[i]*lda+icol[j];
                memcpy(result+dest*esize, A->data+src*esize, esize);
            }
    }
    else
    {
        lda = A->shape[0];
        #pragma omp parallel for private(dest, src, i, j)
        for(i = 0; i < nrows; i++)
            for(j = 0; j < ncols; j++)
            {
                dest = j*nrows+i;
                src = icol[j]*lda+irow[i];
                memcpy(result+dest*esize, A->data+src*esize, esize);
            }
    }
    return 0;
}

int STARS_Problem_from_array(STARS_Problem **P, Array *A, char symm)
// Generate STARS_Problem with a given array and flag if it is symmetric
{
    if(P == NULL)
    {
        STARS_error("STARS_Problem_from_array", "invalid value of `P`");
        return 1;
    }
    if(A == NULL)
    {
        STARS_error("STARS_Problem_from_array", "invalid value of `A`");
        return 1;
    }
    if(A->ndim < 2)
    {
        STARS_error("STARS_Problem_from_array", "`A` should be at least "
                "2-dimensional");
        return 1;
    }
    if(symm != 'S' && symm != 'N')
    {
        STARS_error("STARS_Problem_from_array", "invalid value of `symm`");
        return 1;
    }
    Array *A2 = A;
    int info;
    if(A->order == 'C')
    {
        STARS_warning("STARS_Problem_from_array", "A->order is 'C', creating "
                "copy of array with layout in Fortran style ('F'-order). It "
                "makes corresponding matrix non-freeable");
        info = Array_new_copy(&A2, A, 'F');
        if(info != 0)
            return info;
    }
    info = STARS_Problem_new(P, A->ndim, A->shape, symm, A->dtype, A2, A2,
            _matrix_kernel, "Problem from matrix");
    return info;
}

int STARS_Problem_to_array(STARS_Problem *P, Array **A)
// Compute matrix/array, corresponding to the P
{
    if(P == NULL)
    {
        STARS_error("STARS_Problem_to_array", "invalid value of `P`");
        return 1;
    }
    if(A == NULL)
    {
        STARS_error("STARS_Problem_to_array", "invalid value of `A`");
        return 1;
    }
    int ndim = P->ndim, i, nrows = P->shape[0], ncols = P->shape[ndim-1], info;
    int *irow = malloc(nrows*sizeof(*irow));
    if(irow == NULL)
    {
        STARS_error("STARS_Problem_to_array", "malloc() failed");
        return 1;
    }
    int *icol = malloc(ncols*sizeof(*icol));
    if(icol == NULL)
    {
        STARS_error("STARS_PRoblem_to_array", "malloc() failed");
        return 1;
    }
    for(i = 0; i < nrows; i++)
        irow[i] = i;
    for(i = 0; i < ncols; i++)
        icol[i] = i;
    info = STARS_Problem_get_block(P, nrows, ncols, irow, icol, A);
    free(irow);
    free(icol);
    return info;
}
