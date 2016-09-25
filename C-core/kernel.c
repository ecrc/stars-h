#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "blr.h"
#include "kernel.h"
#include "cblas.h"
#include "misc.h"

/*
double whittleKernel(double x, double beta)
{
    return  x/beta < 1e-4 ? 1.0 : cyl_bessel_k(1, x/beta);
}
*/

void block_exp_kernel(int rows, int cols, int *row, int *col,
        STARS_ssdata *row_data, STARS_ssdata *col_data, double *result)
{
    // Block kernel for spatial statistics
    // Returns exp^{-r/beta}, where r is a distance between particles in 2D
    int i, j;
    double tmp, dist, beta = -row_data->beta;
    double *x = row_data->point;
    double *y = row_data->point+row_data->count;
    for(j = 0; j < cols; j++)
        for(i = 0; i < rows; i++)
        {
            tmp = x[row[i]]-x[col[j]];
            dist = tmp*tmp;
            tmp = y[row[i]]-y[col[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            result[j*rows+i] = exp(dist);
        }
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

int compare_int(int *a, int *b)
{
    return *a-*b;
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

block_func *STARS_get_kernel(char *name)
{
    // Returns pointer to a kernel by a given problem name
    if(strcmp(name, "spatial") == 0)
    {
        return block_exp_kernel;
    }
    else if(strcmp(name, "synth") == 0)
    {
        return block_synth_kernel;
    }
    else
    {
        printf("Wrong parameter to STARS_get_kernel\n");
        return NULL;
    }
}

void block_synth_kernel(int rows, int cols, int *row, int *col,
        STARS_synthdata *row_data, STARS_synthdata *col_data, double *result)
{
    // Block kernel for matrix USV
    int i, j, k;
    int bsize = row_data->bsize;
    int bcount = row_data->bcount;
    double *U = row_data->U;
    double *S = row_data->S;
    double *V = row_data->V;
    double *tmp_buf = (double *)malloc(bsize*sizeof(double));
    double *ptr;
    for(j = 0; j < cols; j++)
        for(i = 0; i < rows; i++)
        {
            ptr = U+row[i];
            for(k = 0; k < bsize; k++)
                tmp_buf[k] = ptr[k*bcount]*S[k];
            result[j*rows+i] = cblas_ddot(bsize, tmp_buf, 1,
                    V+col[j]*bsize, 1);
        }
    free(tmp_buf);
}

STARS_synthdata *STARS_gen_synthdata(int rows, int cols, int brows, int bcols,
        int *brow_start, int *bcol_start, int *brow_size, int *bcol_size)
{
    int i, j, n;
    STARS_synthdata *data = (STARS_synthdata *)malloc(sizeof(STARS_synthdata));
    int size = 0, max_size = 0;
    for(i = 0; i < brows; i++)
    {
        n = brow_size[i];
        size += n*n;
        max_size = max_size > n ? max_size : n;
    }
    data->U = (double *)malloc(size*sizeof(double));
    size = 0;
    for(i = 0; i < brows; i++)
    {
        n = brow_size[i];
        dmatrix_randn(n, n, data->U+size);
        dmatrix_qr(n, n, data->U+size, data->U+size, NULL);
        size += n*n;
    }
    size = 0;
    for(i = 0; i < bcols; i++)
    {
        size += bcol_size[i]*bcol_size[i];
    }
    data->V = (double *)malloc(size*sizeof(double));
    size = 0;
    for(i = 0; i < bcols; i++)
    {
        n = bcol_size[i];
        dmatrix_randn(n, n, data->V+size);
        dmatrix_qr(n, n, data->V+size, data->V+size, NULL);
        size += n*n;
    }
    data->S = (double *)malloc(max_size*sizeof(double));
    return data;
}
