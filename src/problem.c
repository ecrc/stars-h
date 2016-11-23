#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <complex.h>
#include "stars.h"

STARS_Problem *STARS_Problem_init(size_t ndim, size_t *shape, char symm,
        char dtype, void *row_data, void *col_data, block_kernel kernel,
        char *name)
// Init for STARS_Problem instance
// Parameters:
//   ndim: dimensionality of corresponding array. Equal 2+dimensionality of
//     kernel
//   shape: shape of corresponding array. shape[1:ndim-2] is equal to shape of
//     kernel
//   symm: 'S' for summetric problem, 'N' for nonsymmetric problem. Symmetric
//     problem require symmetric kernel and equality of row_data and col_data
//   dtype: data type of the problem. Equal to 's', 'd', 'c' or 'z' as in
//     LAPACK routines. Stands for data type of an element of a kernel.
//   row_data: pointer to some structure of physical data for rows
//   col_data: pointer to some srructure of physical data for columns
//   kernel: pointer to a function of interaction. More on this is written
//     somewhere else.
//   name: string, containgin name of the problem. Used only to print
//     information about structure problem.
//  Returns:
//    STARS_Problem *: pointer to structure problem with proper filling of all
//    the fields of structure.
{
    if(ndim < 2)
    {
        fprintf(stderr, "Parameter 1 in STARS_Problem_init is wrong (should be"
                " at least 2\n");
        return NULL;
    }
    size_t i, dtype_size = 0, entry_size = dtype_size;
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
        fprintf(stderr, "Wrong parameter 4 in STARS_Problem_init\n");
        return NULL;
    }
    for(i = 1; i < ndim-1; i++)
        entry_size *= shape[i];
    STARS_Problem *problem = malloc(sizeof(*problem));
    problem->ndim = ndim;
    problem->shape = malloc(ndim*sizeof(*problem->shape));
    memcpy(problem->shape, shape, ndim*sizeof(*problem->shape));
    problem->symm = symm;
    problem->dtype = dtype;
    problem->dtype_size = dtype_size;
    problem->entry_size = entry_size;
    problem->row_data = row_data;
    problem->col_data = col_data;
    problem->kernel = kernel;
    problem->name = malloc(strlen(name)+1);
    strcpy(problem->name, name);
    return problem;
}

void STARS_Problem_free(STARS_Problem *problem)
// Free memory, consumed by data buffers of data
{
    if(problem == NULL)
    {
        fprintf(stderr, "STARS_Problem instance is NOT initialized\n");
        return;
    }
    free(problem->shape);
    free(problem->name);
    free(problem);
}

void STARS_Problem_info(STARS_Problem *problem)
// Print some info about Problem
{
    printf("<STARS_Problem at %p, name \"%s\", shape (%zu",
            problem, problem->name, problem->shape[0]);
    for(size_t i = 1; i < problem->ndim; i++)
        printf(",%zu", problem->shape[i]);
    printf("), '%c' dtype, '%c' symmetric>\n", problem->dtype, problem->symm);
}

Array *STARS_Problem_get_block(STARS_Problem *problem, size_t nrows,
        size_t ncols, size_t *irow, size_t *icol)
// Get submatrix on given rows and columns (rows=first dimension, columns=last
// dimension)
{
    size_t ndim = problem->ndim;
    size_t *shape = malloc(ndim*sizeof(*shape));
    shape[0] = nrows;
    shape[ndim-1] = ncols;
    memcpy(shape+1, problem->shape+1, (ndim-2)*sizeof(*shape));
    Array *array = Array_new(ndim, shape, problem->dtype, 'F');
    problem->kernel(nrows, ncols, irow, icol, problem->row_data,
            problem->col_data, array->buffer);
    return array;
}

static int _matrix_kernel(size_t nrows, size_t ncols, size_t *irow,
        size_t *icol, void *row_data, void *col_data, void *result)
{
    Array *data = row_data;
    size_t esize = data->dtype_size;
    size_t i, j, dest, src, lda;
    for(i = 1; i < data->ndim-1; i++)
        esize *= data->shape[i];
    if(data->order == 'C')
    {
        lda = data->shape[data->ndim-1];
        #pragma omp parallel for private(dest, src, i, j)
        for(i = 0; i < nrows; i++)
            for(j = 0; j < ncols; j++)
            {
                dest = j*nrows+i;
                src = irow[i]*lda+icol[j];
                memcpy(result+dest*esize, data->buffer+src*esize, esize);
            }
    }
    else
    {
        lda = data->shape[0];
        #pragma omp parallel for private(dest, src, i, j)
        for(i = 0; i < nrows; i++)
            for(j = 0; j < ncols; j++)
            {
                dest = j*nrows+i;
                src = icol[j]*lda+irow[i];
                memcpy(result+dest*esize, data->buffer+src*esize, esize);
            }
    }
    return 0;
}

STARS_Problem *STARS_Problem_from_array(Array *array, char symm)
// Generate STARS_Problem with a given array and flag if it is symmetric
{
    if(array->ndim < 2)
    {
        fprintf(stderr, "Parameter 1 to STARS_Problem_from_array should be at "
                "least 2-dimensional\n");
        return NULL;
    }
    if(symm != 'S' && symm != 'N')
    {
        fprintf(stderr, "Parameter 2 to STARS_Problem_from_array should be "
                "'S' or 'N'\n");
        return NULL;
    }
    Array *array2 = array;
    if(array->order == 'C')
    {
        printf("Order of input array in STARS_Problem_from_array is 'C', "
                "creating copy of array with layout in Fortran style "
                "('F'-order)\n");
        array2 = Array_copy(array, 'F');
    }
    STARS_Problem *problem = STARS_Problem_init(array->ndim, array->shape,
            symm, array->dtype, array2, array2, _matrix_kernel,
            "Problem from matrix");
    return problem;
}

Array *STARS_Problem_to_array(STARS_Problem *problem)
// Compute matrix/array, corresponding to the problem
{
    size_t ndim = problem->ndim, i;
    size_t nrows = problem->shape[0];
    size_t ncols = problem->shape[ndim-1];
    size_t *irow = malloc(nrows*sizeof(*irow));
    size_t *icol = malloc(ncols*sizeof(*icol));
    for(i = 0; i < nrows; i++)
        irow[i] = i;
    for(i = 0; i < ncols; i++)
        icol[i] = i;
    Array *array = STARS_Problem_get_block(problem, nrows, ncols, irow, icol);
    free(irow);
    free(icol);
    return array;
}
