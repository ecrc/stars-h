#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "stars.h"
#include "stars-misc.h"
#include "stars-spatial.h"


int STARS_ssdata_block_exp_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result)
{
    // Block kernel for spatial statistics
    // Returns exp^{-r/beta}, where r is a distance between particles in 2D
    int i, j;
    STARS_ssdata *data = row_data;
    double tmp, dist, beta = -data->beta;
    double *x = data->point, *y = x+data->count;
    double *buffer = result;
    #pragma omp parallel for private(tmp, dist, i, j)
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            buffer[j*(size_t)nrows+i] = exp(dist);
        }
    return 0;
}

uint32_t Part1By1(uint32_t x)
{
  x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}

uint32_t Compact1By1(uint32_t x)
{
  x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

uint32_t EncodeMorton2(uint32_t x, uint32_t y) { return (Part1By1(y) << 1) + Part1By1(x); }
uint32_t DecodeMorton2X(uint32_t code) { return Compact1By1(code >> 0); }
uint32_t DecodeMorton2Y(uint32_t code) { return Compact1By1(code >> 1); }

int compare_int(const void *a, const void *b)
{
    return *(int *)a-*(int *)b;
}

void zsort(int n, double *points)
{
    // Some sorting, required by spatial statistics code
    int i;
    uint16_t x, y;
    uint32_t z[n];
    for(i = 0; i < n; i++)
    {
        x = (uint16_t)(points[i]*(double)UINT16_MAX + .5);
        y = (uint16_t)(points[i+n]*(double)UINT16_MAX + .5);
        z[i] = EncodeMorton2(x, y);
    }
    qsort(z, n, sizeof(uint32_t), compare_int);
    for(i = 0; i < n; i++)
    {
        points[i] = (double)DecodeMorton2X(z[i])/(double)UINT16_MAX;
        points[i+n] = (double)DecodeMorton2Y(z[i])/(double)UINT16_MAX;
    }
}

void gen_points_old(int n, double *points)
{
    int i, j;
    double *A = points+n*n;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            points[i*n+j] = (j+0.5+0.4*rand()/RAND_MAX)/n;
            A[i*n+j] = (i+0.5+0.4*rand()/RAND_MAX)/n;
        }
    }
}

void gen_ss_block_points(int m, int n, int block_size, double *points)
{
    int i, j, k;
    size_t npoints = (size_t)m*(size_t)n*(size_t)block_size, ind = 0;
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

STARS_ssdata *STARS_gen_ssdata(int row_blocks, int col_blocks,
        int block_size, double beta)
{
    size_t n = (size_t)row_blocks*(size_t)col_blocks*(size_t)block_size;
    STARS_ssdata *data = malloc(sizeof(*data));
    data->point = malloc(2*n*sizeof(*data->point));
    gen_ss_block_points(row_blocks, col_blocks, block_size, data->point);
    data->count = n;
    data->beta = beta;
    return data;
}

STARS_ssdata *STARS_gen_ssdata2(int n, double beta)
{
    STARS_ssdata *data = malloc(sizeof(*data));
    data->point = malloc(2*n*n*sizeof(*data->point));
    gen_points_old(n, data->point);
    zsort(n*n, data->point);
    data->count = n*n;
    data->beta = beta;
    return data;
}

void STARS_ssdata_free(STARS_ssdata *data)
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
STARS_Problem *STARS_gen_ssproblem(int row_blocks, int col_blocks,
        int block_size, double beta)
{
    STARS_ssdata *data = STARS_gen_ssdata(row_blocks, col_blocks, block_size,
            beta);
    int shape[2] = {data->count, data->count};
    return STARS_Problem_init(2, shape, 'S', 'd', data, data,
            block_exp_kernel_noalloc, "Spatial Statistics problem");
}

STARS_BLR *STARS_gen_ss_blrformat(int row_blocks, int col_blocks,
        int block_size, double beta)
{
    int i, block_count = row_blocks*col_blocks, n = block_size*block_count;
    STARS_BLR *blr = (STARS_BLR *)malloc(sizeof(STARS_BLR));
    blr->problem = STARS_gen_ssproblem(row_blocks, col_blocks, block_size,
            beta);
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
