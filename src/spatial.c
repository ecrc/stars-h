#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "stars-misc.h"
#include "stars-spatial.h"

void *block_exp_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result)
{
    // Block kernel for spatial statistics
    // Returns exp^{-r/beta}, where r is a distance between particles in 2D
    int i, j;
    STARS_ssdata *rdata = (STARS_ssdata *)row_data;
    STARS_ssdata *cdata = (STARS_ssdata *)col_data;
    double tmp, dist, beta = -rdata->beta;
    double *x = rdata->point;
    double *y = rdata->point+rdata->count;
    double *out = (double *)result;
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            out[j*nrows+i] = exp(dist);
        }
    return NULL;
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

void gen_points(int n, double *points)
{
    int i, j;
    double *A = points+n*n;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            points[i*n+j] = (j+0.5+0.4*randn())/n;
            A[i*n+j] = (i+0.5+0.4*randn())/n;
        }
    }
}

void *STARS_gen_ssdata(int n, double beta)
{
    STARS_ssdata *data = (STARS_ssdata *)malloc(sizeof(STARS_ssdata));
    data->point = (double *)malloc(2*n*n*sizeof(double));
    gen_points(n, data->point);
    zsort(n*n, data->point);
    data->count = n*n;
    data->beta = beta;
    return (void *)data;
}

STARS_Problem *STARS_gen_ssproblem(int n, double beta)
{
    STARS_Problem *problem = (STARS_Problem *)malloc(sizeof(STARS_Problem));
    problem->nrows = n*n;
    problem->ncols = n*n;
    problem->symm = 'S';
    problem->dtype = 'd';
    problem->row_data = STARS_gen_ssdata(n, beta);
    problem->col_data = problem->row_data;
    problem->kernel = block_exp_kernel;
    return problem;
}

STARS_BLR *STARS_gen_ss_blrformat(int n, double beta)
{
    int i;
    STARS_BLR *blr = (STARS_BLR *)malloc(sizeof(STARS_BLR));
    blr->problem = STARS_gen_ssproblem(n, beta);
    blr->symm = 'S';
    blr->nrows = blr->problem->nrows;
    blr->ncols = blr->nrows;
    blr->row_order = (int *)malloc(blr->nrows*sizeof(int));
    blr->col_order = blr->row_order;
    for(i = 0; i < blr->nrows; i++)
    {
        blr->row_order[i] = i;
    }
    blr->nbrows = n;
    blr->nbcols = n;
    blr->ibrow_start = (int *)malloc(n*sizeof(int));
    blr->ibcol_start = blr->ibrow_start;
    blr->ibrow_size = (int *)malloc(n*sizeof(int));
    blr->ibcol_size = blr->ibrow_size;
    for(i = 0; i < n; i++)
    {
        blr->ibrow_start[i] = i*n;
        blr->ibrow_size[i] = n;
    }
    return blr;
}
