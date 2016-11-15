#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "stars.h"
#include "stars-electrostatics.h"


int STARS_esdata_block_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result)
{
    // Block kernel for electrostatics
    // Returns r^-1, where r is a distance between particles in 2D
    int i, j;
    STARS_esdata *data = row_data;
    double tmp, dist;
    double *x = data->point, *y = x+data->count;
    double *buffer = result;
    #pragma omp parallel for private(tmp, dist, i, j)
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

void gen_es_block_points(int m, int n, int block_size, double *points)
{
    int i, j, k, ind = 0;
    int npoints = m*n*block_size;
    double *x = points, *y = points+npoints;
    double noise_var = 1., rand_max = RAND_MAX;;
    for(i = 0; i < m; i++)
        for(j = 0; j < n; j++)
            for(k = 0; k < block_size; k++)
            {
                x[ind] = (j+noise_var*rand()/rand_max)/n;
                y[ind] = (i+noise_var*rand()/rand_max)/m;
                ind++;
            }
}

STARS_esdata *STARS_gen_esdata(int row_blocks, int col_blocks, int block_size)
{
    int n = row_blocks*col_blocks*block_size;
    STARS_esdata *data = malloc(sizeof(*data));
    data->point = malloc(2*n*sizeof(double));
    gen_es_block_points(row_blocks, col_blocks, block_size, data->point);
    data->count = n;
    return data;
}

void STARS_esdata_free(STARS_esdata *data)
{
    if(data == NULL)
    {
        fprintf(stderr, "Data for spatial statistics problem was not "
                "generated\n");
        return;
    }
    free(data->point);
    free(data);
}
/*
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
    //!blr->nrows = blr->problem->nrows;
    //!blr->ncols = blr->nrows;
    blr->row_pivot = (int *)malloc(blr->nrows*sizeof(int));
    blr->col_pivot = blr->row_pivot;
    for(i = 0; i < blr->nrows; i++)
    {
        blr->row_pivot[i] = i;
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
*/
