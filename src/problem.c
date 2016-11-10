#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "stars.h"

STARS_Problem *STARS_Problem_init(int nrows, int ncols, char symm, char dtype,
        void *row_data, void *col_data, block_kernel kernel, char *type)
{
    STARS_Problem *problem = (STARS_Problem *)malloc(sizeof(STARS_Problem));
    problem->nrows = nrows;
    problem->ncols = ncols;
    problem->symm = symm;
    problem->dtype = dtype;
    problem->row_data = row_data;
    problem->col_data = col_data;
    problem->kernel = kernel;
    problem->type = type;
    return problem;
}

void STARS_Problem_info(STARS_Problem *problem)
{
    printf("<STARS_Problem object at %p, %dx%d matrix, %s type>\n",
            (char *)problem, problem->nrows, problem->ncols, problem->type);
}

void *STARS_Problem_dotvec(char side, STARS_Problem *problem, void *vec,
        char dtype)
{
    switch(dtype)
    {
        case 'd':
            break;
        case 's':
        case 'c':
        case 'z':
            break;
    }
    return NULL;
}

int _matrix_kernel(int nrows, int ncols, int *irow, int *icol, void *rdata,
        void *cdata, void *result)
{
    Array *data = rdata;
    size_t dsize = data->dtype_size;
    int i, j, dest, src, lda;
    if(data->order == 'C')
    {
        lda = data->shape[1];
        for(i = 0; i < nrows; i++)
            for(j = 0; j < ncols; j++)
            {
                dest = j*nrows+i;
                src = irow[i]*lda+icol[j];
                memcpy(result+dest*dsize, data->buffer+src*dsize, dsize);
            }
    }
    else
    {
        lda = data->shape[0];
        for(i = 0; i < nrows; i++)
            for(j = 0; j < ncols; j++)
            {
                dest = j*nrows+i;
                src = icol[j]*lda+irow[i];
                memcpy(result+dest*dsize, data->buffer+src*dsize, dsize);
            }
    }
}

STARS_Problem *STARS_Problem_from_array(Array *data)
{
    STARS_Problem *problem = (STARS_Problem *)malloc(sizeof(STARS_Problem));
    if(data->ndim != 2)
    {
        fprintf(stderr, "Input array should be 2-dimensional\n");
        return NULL;
    }
    problem->nrows = data->shape[0];
    problem->ncols = data->shape[1];
    problem->symm = 'N';
    problem->dtype = data->dtype;
    problem->dtype_size = data->dtype_size;
    problem->row_data = data;
    problem->col_data = data;
    problem->kernel = _matrix_kernel;
    problem->type = "Array";
    return problem;
}
