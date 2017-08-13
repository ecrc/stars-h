/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/problem.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-05-21
 * */

#include "common.h"
#include "starsh.h"

int starsh_problem_new(STARSH_problem **P, int ndim, int *shape, char symm,
        char dtype, void *row_data, void *col_data, STARSH_kernel kernel,
        char *name)
//! Init for STARSH_problem instance.
/*! @ingroup problem
 * @param[out] P: Address of pointer to `STARSH_problem` object.
 * @param[in] ndim: Dimensionality of corresponding array. Equal `2` plus
 *     dimensionality of kernel.
 * @param[in] shape: Shape of corresponding array. Subarray `shape[1:ndim-2]`
 *     is equal to shape of kernel.
 * @param[in] symm: 'S' for summetric problem, 'N' for nonsymmetric. Symmetric
 *     problem requires symmetric kernel and equality of `row_data` and
 *     `col_data`.
 * @param[in] dtype: Data type of the problem. Equal to `'s'`, `'d'`, `'c'` or
 *     `'z'` as in LAPACK routines.
 * @param[in] row_data: Pointer to some structure of physical data for rows.
 * @param[in] col_data: Pointer to some structure of physical data for columns.
 * @param[in] kernel: Pointer to a function of interaction.
 * @param[in] name: String, containing name of the problem.
 * @return Error code.
 * */
{
    if(P == NULL)
    {
        STARSH_ERROR("invalid value of `P`");
        return 1;
    }
    if(ndim < 2)
    {
        STARSH_ERROR("invalid value of `ndim`");
        return 1;
    }
    if(shape == NULL)
    {
        STARSH_ERROR("invalid value of `shape`");
        return 1;
    }
    if(kernel == NULL)
    {
        STARSH_ERROR("invalud value of `kernel`");
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
        STARSH_ERROR("invalid value of `dtype`");
        return 1;
    }
    size_t entry_size = dtype_size;
    for(i = 1; i < ndim-1; i++)
        entry_size *= shape[i];
    STARSH_MALLOC(*P, 1);
    STARSH_problem *P2 = *P;
    P2->ndim = ndim;
    STARSH_MALLOC(P2->shape, ndim);
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
        STARSH_MALLOC(P2->name, strlen(name)+1);
        strcpy(P2->name, name);
    }
    return 0;
}

int starsh_problem_free(STARSH_problem *P)
//! Free fields and structure of the problem.
//! @ingroup problem
{
    if(P == NULL)
    {
        STARSH_ERROR("invalid value of `P`");
        return 1;
    }
    free(P->shape);
    if(P->name != NULL)
        free(P->name);
    free(P);
    return 0;
}

int starsh_problem_info(STARSH_problem *P)
//! Print some info about problem.
//! @ingroup problem
{
    if(P == NULL)
    {
        STARSH_ERROR("invalid value of `P`");
        return 1;
    }
    printf("<STARS_Problem at %p, name \"%s\", shape (%d", P, P->name,
            P->shape[0]);
    for(int i = 1; i < P->ndim; i++)
        printf(",%d", P->shape[i]);
    printf("), '%c' dtype, '%c' symmetric>\n", P->dtype, P->symm);
    return 0;
}

int starsh_problem_get_block(STARSH_problem *P, int nrows, int ncols,
        int *irow, int *icol, Array **A)
//! Get submatrix on given rows and columns.
//! @ingroup problem
/*! Rows correspond to the first dimension and columns correspond to the
 * last dimension. */
{
    if(P == NULL)
    {
        STARSH_ERROR("invalid value of `P`");
        return 1;
    }
    if(irow == NULL)
    {
        STARSH_ERROR("invalid value of `irow`");
        return 1;
    }
    if(icol == NULL)
    {
        STARSH_ERROR("invalid value of `icol`");
        return 1;
    }
    int ndim = P->ndim, info;
    if(nrows < 0)
    {
        STARSH_ERROR("invalid value of `nrows`");
        return 1;
    }
    if(ncols < 0)
    {
        STARSH_ERROR("invalid value of `ncols`");
        return 1;
    }
    int *shape;
    STARSH_MALLOC(shape, ndim);
    shape[0] = nrows;
    shape[ndim-1] = ncols;
    memcpy(shape+1, P->shape+1, (ndim-2)*sizeof(*shape));
    info = array_new(A, ndim, shape, P->dtype, 'F');
    if(info != 0)
        return info;
    P->kernel(nrows, ncols, irow, icol, P->row_data, P->col_data,
            (*A)->data);
    return 0;
}

static void _matrix_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result)
//! Kernel for problems, defined by dense matrices.
//! @ingroup problem
{
    Array *A = row_data;
    size_t esize = A->dtype_size;
    size_t i, j, dest, src, lda;
    for(i = 1; i < A->ndim-1; i++)
        esize *= A->shape[i];
    if(A->order == 'C')
    {
        lda = A->shape[A->ndim-1];
        //#pragma omp parallel for private(dest, src, i, j)
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
        //#pragma omp parallel for private(dest, src, i, j)
        for(i = 0; i < nrows; i++)
            for(j = 0; j < ncols; j++)
            {
                dest = j*nrows+i;
                src = icol[j]*lda+irow[i];
                memcpy(result+dest*esize, A->data+src*esize, esize);
            }
    }
}

int starsh_problem_from_array(STARSH_problem **P, Array *A, char symm)
//! Create STARSH_problem instance, based on dense array.
//! @ingroup problem
{
    if(P == NULL)
    {
        STARSH_ERROR("invalid value of `P`");
        return 1;
    }
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    if(A->ndim < 2)
    {
        STARSH_ERROR("`A` should be at least 2-dimensional");
        return 1;
    }
    if(symm != 'S' && symm != 'N')
    {
        STARSH_ERROR("invalid value of `symm`");
        return 1;
    }
    Array *A2 = A;
    int info;
    if(A->order == 'C')
    {
        STARSH_WARNING("A->order is 'C', creating "
                "copy of array with layout in Fortran style ('F'-order). It "
                "makes corresponding matrix non-freeable");
        info = array_new_copy(&A2, A, 'F');
        if(info != 0)
            return info;
    }
    info = starsh_problem_new(P, A->ndim, A->shape, symm, A->dtype, A2, A2,
            _matrix_kernel, "Problem from matrix");
    return info;
}

int starsh_problem_to_array(STARSH_problem *P, Array **A)
//! Generate dense array by a given problem.
//! @ingroup problem
{
    if(P == NULL)
    {
        STARSH_ERROR("invalid value of `P`");
        return 1;
    }
    if(A == NULL)
    {
        STARSH_ERROR("invalid value of `A`");
        return 1;
    }
    int ndim = P->ndim, i, nrows = P->shape[0], ncols = P->shape[ndim-1], info;
    int *irow, *icol;
    STARSH_MALLOC(irow, nrows);
    STARSH_MALLOC(icol, ncols);
    for(i = 0; i < nrows; i++)
        irow[i] = i;
    for(i = 0; i < ncols; i++)
        icol[i] = i;
    info = starsh_problem_get_block(P, nrows, ncols, irow, icol, A);
    free(irow);
    free(icol);
    return info;
}
