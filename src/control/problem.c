/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/control/problem.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include "common.h"
#include "starsh.h"

int starsh_problem_new(STARSH_problem **problem, int ndim, STARSH_int *shape,
        char symm, char dtype, void *row_data, void *col_data,
        STARSH_kernel *kernel, char *name)
//! Init @ref STARSH_problem object.
/*! Unlike all other *_new() functions, this function creates copy of `shape`
 * to store internally. This is done to avoid clearing memory of static
 * objects, defined like `STARSH_int shape[2] = {10, 20}`. Number of dimensions
 * must be 2 or greater. If `ndim = 2`, then corresponding kernel is scalar. If
 * `ndim > 2`, then corresponding kernel returns `(ndim-2)`-dimensional tensor.
 *
 * @param[out] problem: Address of pointer to @ref STARSH_problem object.
 * @param[in] ndim: Dimensionality of corresponding array. Equal to `2` plus
 *      dimensionality of kernel.
 * @param[in] shape: Shape of corresponding array. Subarray `shape[1:ndim-2]`
 *      is equal to shape of kernel.
 * @param[in] symm: 'S' for summetric problem, 'N' for nonsymmetric. Symmetric
 *      problem requires symmetric kernel and equality of `row_data` and
 *      `col_data`.
 * @param[in] dtype: Data type of the problem. Equal to 's', 'd', 'c' or 'z'
 *      as in LAPACK routines.
 * @param[in] row_data: Pointer to some structure of physical data for rows.
 * @param[in] col_data: Pointer to some structure of physical data for columns.
 * @param[in] kernel: Pointer to a function of interaction.
 * @param[in] name: String, containing name of the problem.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup problem
 * */
{
    if(problem == NULL)
    {
        STARSH_ERROR("Invalid value of `problem`");
        return STARSH_WRONG_PARAMETER;
    }
    if(ndim < 2)
    {
        STARSH_ERROR("Invalid value of `ndim`");
        return STARSH_WRONG_PARAMETER;
    }
    if(shape == NULL)
    {
        STARSH_ERROR("Invalid value of `shape`");
        return STARSH_WRONG_PARAMETER;
    }
    if(kernel == NULL)
    {
        STARSH_ERROR("Invalid value of `kernel`");
        return STARSH_WRONG_PARAMETER;
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
        STARSH_ERROR("Invalid value of `dtype`");
        return STARSH_WRONG_PARAMETER;
    }
    size_t entry_size = dtype_size;
    for(i = 1; i < ndim-1; i++)
        entry_size *= shape[i];
    STARSH_problem *P;
    STARSH_MALLOC(P, 1);
    *problem = P;
    P->ndim = ndim;
    STARSH_MALLOC(P->shape, ndim);
    memcpy(P->shape, shape, ndim*sizeof(*P->shape));
    P->symm = symm;
    P->dtype = dtype;
    P->dtype_size = dtype_size;
    P->entry_size = entry_size;
    P->row_data = row_data;
    P->col_data = col_data;
    P->kernel = kernel;
    P->name = NULL;
    if(name != NULL)
    {
        STARSH_MALLOC(P->name, strlen(name)+1);
        strcpy(P->name, name);
    }
    return STARSH_SUCCESS;
}

void starsh_problem_free(STARSH_problem *problem)
//! Free @ref STARSH_problem object.
//! @ingroup problem
{
    if(problem == NULL)
        return;
    free(problem->shape);
    if(problem->name != NULL)
        free(problem->name);
    free(problem);
}

void starsh_problem_info(STARSH_problem *problem)
//! Print short info about @ref STARSH_problem object.
//! @ingroup problem
{
    if(problem == NULL)
        return;
    STARSH_problem *P = problem;
    printf("<STARS_Problem at %p, name \"%s\", shape (%d", P, P->name,
            P->shape[0]);
    for(int i = 1; i < P->ndim; i++)
        printf(",%d", P->shape[i]);
    printf("), '%c' dtype, '%c' symmetric>\n", P->dtype, P->symm);
}

int starsh_problem_get_block(STARSH_problem *problem, int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, Array **A)
//! Get submatrix on given rows and columns.
/*! Rows correspond to the first dimension and columns correspond to the
 * last dimension.
 *
 * @param[in] problem: Pointer to @ref STARSH_problem object.
 * @param[in] nrows: Number of rows.
 * @param[in] ncols: Number of columns.
 * @param[in] irow: Indexes of rows.
 * @param[in] icol: Indexes of columns.
 * @param[out] A: Address of pointer to @ref array object.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup problem
 * */
{
    STARSH_problem *P = problem;
    if(problem == NULL)
    {
        STARSH_ERROR("Invalid value of `parameter`");
        return STARSH_WRONG_PARAMETER;
    }
    if(irow == NULL)
    {
        STARSH_ERROR("Invalid value of `irow`");
        return STARSH_WRONG_PARAMETER;
    }
    if(icol == NULL)
    {
        STARSH_ERROR("Invalid value of `icol`");
        return STARSH_WRONG_PARAMETER;
    }
    int ndim = P->ndim, info;
    if(nrows < 0)
    {
        STARSH_ERROR("Invalid value of `nrows`");
        return STARSH_WRONG_PARAMETER;
    }
    if(ncols < 0)
    {
        STARSH_ERROR("Invalid value of `ncols`");
        return STARSH_WRONG_PARAMETER;
    }
    int *shape;
    STARSH_MALLOC(shape, ndim);
    shape[0] = nrows;
    shape[ndim-1] = ncols;
    for(int i = 1; i < ndim-1; i++)
    {
        shape[i] = problem->shape[i];
    }
    info = array_new(A, ndim, shape, problem->dtype, 'F');
    if(info != 0)
        return info;
    problem->kernel(nrows, ncols, irow, icol, problem->row_data,
            problem->col_data, (*A)->data, nrows);
    return STARSH_SUCCESS;
}

static void _matrix_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld)
//! Kernel for problems, defined by dense matrices.
//! @ingroup problem
{
    Array *A = row_data;
    size_t esize = A->dtype_size;
    STARSH_int i, j;
    size_t dest, src, lda;
    for(int i = 1; i < A->ndim-1; i++)
        esize *= A->shape[i];
    if(A->order == 'C')
    {
        lda = A->shape[A->ndim-1];
        //#pragma omp parallel for private(dest, src, i, j)
        for(i = 0; i < nrows; i++)
            for(j = 0; j < ncols; j++)
            {
                dest = j*(size_t)ld+i;
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
                dest = j*(size_t)ld+i;
                src = icol[j]*lda+irow[i];
                memcpy(result+dest*esize, A->data+src*esize, esize);
            }
    }
}

int starsh_problem_from_array(STARSH_problem **problem, Array *A, char symm)
//! Create STARSH_problem instance, based on dense array.
/*! If @ref array `A` is sorted in C order, then temporary @ref array object
 * will be created as a copy of input `A`, but in Fortran order. There will be
 * no way to free that temporary @ref array object.
 *
 * @param[out] problem: Address of pointer to @ref STARSH_problem object.
 * @param[in] A: Array.
 * @param[in] symm: 'S' if @ref array `A` is symmetric or 'N' otherwise.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup problem
 * */
{
    if(problem == NULL)
    {
        STARSH_ERROR("Invalid value of `problem`");
        return STARSH_WRONG_PARAMETER;
    }
    if(A == NULL)
    {
        STARSH_ERROR("Invalid value of `A`");
        return STARSH_WRONG_PARAMETER;
    }
    if(A->ndim < 2)
    {
        STARSH_ERROR("`A` should be at least two-dimensional");
        return STARSH_WRONG_PARAMETER;
    }
    if(symm != 'S' && symm != 'N')
    {
        STARSH_ERROR("Invalid value of `symm`");
        return STARSH_WRONG_PARAMETER;
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
    STARSH_int shape[A->ndim];
    for(int i = 0; i < A->ndim; i++)
        shape[i] = A->shape[i];
    info = starsh_problem_new(problem, A->ndim, shape, symm, A->dtype, A2,
            A2, _matrix_kernel, "Problem from matrix");
    return info;
}

int starsh_problem_to_array(STARSH_problem *problem, Array **A)
//! Generate dense array by a given problem.
/*! Dense matrix will be created. This function makes it easier to check
 * kernel.
 *
 * @param[in] problem: Pointer to @ref STARSH_problem object.
 * @param[out] A: Address of pointer to @ref array object.
 * @return Error code @ref STARSH_ERRNO.
 * @ingroup problem
 * */
{
    if(problem == NULL)
    {
        STARSH_ERROR("Invalid value of `problem`");
        return STARSH_WRONG_PARAMETER;
    }
    if(A == NULL)
    {
        STARSH_ERROR("Invalid value of `A`");
        return STARSH_WRONG_PARAMETER;
    }
    STARSH_int i;
    int info;
    int ndim = problem->ndim;
    STARSH_int nrows = problem->shape[0];
    STARSH_int ncols = problem->shape[ndim-1];
    STARSH_int *irow, *icol;
    STARSH_MALLOC(irow, nrows);
    STARSH_MALLOC(icol, ncols);
    for(i = 0; i < nrows; i++)
        irow[i] = i;
    for(i = 0; i < ncols; i++)
        icol[i] = i;
    info = starsh_problem_get_block(problem, nrows, ncols, irow, icol, A);
    free(irow);
    free(icol);
    return info;
}
