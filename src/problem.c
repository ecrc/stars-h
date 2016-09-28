#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
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
