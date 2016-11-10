#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "stars.h"
#include "stars-spatial.h"

Array *block_es_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data)
{
    // Block kernel for electrostatics
    // Returns r^-1, where r is a distance between particles in 2D
    int i, j;
    STARS_ssdata *rdata = (STARS_ssdata *)row_data;
    STARS_ssdata *cdata = (STARS_ssdata *)col_data;
    double tmp, dist;
    double *x = rdata->point;
    double *y = rdata->point+rdata->count;
    int shape[2] = {nrows, ncols};
    Array *result = Array_new(2, shape, 'd', 'F');
    double *buffer = result->buffer;
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            if(irow[i] != icol[j])
            {
                tmp = x[irow[i]]-x[icol[j]];
                dist = tmp*tmp;
                tmp = y[irow[i]]-y[icol[j]];
                dist += tmp*tmp;
                buffer[j*nrows+i] = 1./sqrt(dist);
            }
            else
                buffer[j*nrows+i] = 0.;
        }
    return result;
}

int block_es_kernel_noalloc(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result)
{
    // Block kernel for electrostatics
    // Returns r^-1, where r is a distance between particles in 2D
    int i, j;
    STARS_ssdata *rdata = (STARS_ssdata *)row_data;
    STARS_ssdata *cdata = (STARS_ssdata *)col_data;
    double tmp, dist;
    double *x = rdata->point;
    double *y = rdata->point+rdata->count;
    double *buffer = result;
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            if(irow[i] != icol[j])
            {
                tmp = x[irow[i]]-x[icol[j]];
                dist = tmp*tmp;
                tmp = y[irow[i]]-y[icol[j]];
                dist += tmp*tmp;
                buffer[j*nrows+i] = 1./sqrt(dist);
            }
            else
                buffer[j*nrows+i] = 0.;
        }
    return 0;
}

STARS_Problem *STARS_gen_esproblem(int row_blocks, int col_blocks,
        int block_size)
{
    STARS_Problem *problem = STARS_gen_ssproblem(row_blocks, col_blocks,
            block_size, 0);
    problem->kernel = block_es_kernel_noalloc;
    return problem;
}

STARS_BLR *STARS_gen_es_blrformat(int row_blocks, int col_blocks,
        int block_size)
{
    int i, block_count = row_blocks*col_blocks, n = block_size*block_count;
    STARS_BLR *blr = (STARS_BLR *)malloc(sizeof(STARS_BLR));
    blr->problem = STARS_gen_esproblem(row_blocks, col_blocks, block_size);
    blr->symm = 'S';
    blr->nrows = blr->problem->nrows;
    blr->ncols = blr->nrows;
    blr->row_order = (int *)malloc(blr->nrows*sizeof(int));
    blr->col_order = blr->row_order;
    for(i = 0; i < blr->nrows; i++)
    {
        blr->row_order[i] = i;
    }
    blr->nbrows = block_count;
    blr->nbcols = block_count;
    blr->ibrow_start = (int *)malloc(block_count*sizeof(int));
    blr->ibcol_start = blr->ibrow_start;
    blr->ibrow_size = (int *)malloc(block_count*sizeof(int));
    blr->ibcol_size = blr->ibrow_size;
    for(i = 0; i < block_count; i++)
    {
        blr->ibrow_start[i] = i*block_size;
        blr->ibrow_size[i] = block_size;
    }
    return blr;
}
