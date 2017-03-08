#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "starsh.h"
#include "starsh-spatial.h"

static void starsh_ssdata_block_exp_kernel(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result)
/*! Kernel for spatial statistics problem
 *
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 */
{
    // Block kernel for spatial statistics
    // Returns exp^{-r/beta}, where r is a distance between particles in 2D
    int i, j;
    STARSH_ssdata *data = row_data;
    double tmp, dist, beta = -data->beta;
    double *x = data->point, *y = x+data->count;
    double *buffer = result;
    //#pragma omp parallel
    //printf("myid %d\n", omp_get_thread_num());
    //#pragma omp parallel for private(tmp, dist, i, j)
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
}

static uint32_t Part1By1(uint32_t x)
{
  x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}

static uint32_t Compact1By1(uint32_t x)
{
  x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

static uint32_t EncodeMorton2(uint32_t x, uint32_t y) { return (Part1By1(y) << 1) + Part1By1(x); }
static uint32_t DecodeMorton2X(uint32_t code) { return Compact1By1(code >> 0); }
static uint32_t DecodeMorton2Y(uint32_t code) { return Compact1By1(code >> 1); }

static int compare_uint32(const void *a, const void *b)
{
    return *(uint32_t *)a-*(uint32_t *)b;
}

static void zsort(int n, double *points)
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
    qsort(z, n, sizeof(uint32_t), compare_uint32);
    for(i = 0; i < n; i++)
    {
        points[i] = (double)DecodeMorton2X(z[i])/(double)UINT16_MAX;
        points[i+n] = (double)DecodeMorton2Y(z[i])/(double)UINT16_MAX;
    }
}

static void gen_points(int n, double *points)
//! Generate particles in 2-dimensional grid plus random noise.
{
    int i, j;
    double *A = points+n*n;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            points[i*n+j] = (j+0.5-0.4*rand()/(double)RAND_MAX)/n;
            A[i*n+j] = (i+0.5-0.4*rand()/(double)RAND_MAX)/n;
        }
    }
}

void starsh_ssdata_free(STARSH_ssdata *data)
//! Free data.
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

int starsh_gen_ssdata(STARSH_ssdata **data, STARSH_kernel *kernel, int n,
        double beta)
//! Generate spatial statistics data.
/*! @param[out] data: Address of pointer to `STARSH_ssdata` object.
 * @param[out] kernel: Interaction kernel (exponential).
 * @param[in] n: Number of grid steps in one dimension. Total number of
 *     elements will be `n^2`.
 * @param[in] beta: Parameter for kernel.
 * @return Error code.
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("invalid value of `data`");
        return 1;
    }
    if(kernel == NULL)
    {
        STARSH_ERROR("invalid value of `kernel`");
        return 1;
    }
    if(n < 0)
    {
        STARSH_ERROR("invalid value of `n`");
        return 1;
    }
    if(beta <= 0)
    {
        STARSH_ERROR("invalid value iof `beta`");
        return 1;
    }
    *data = malloc(sizeof(**data));
    double *point;
    STARSH_MALLOC(point, 2*(size_t)n*(size_t)n);
    gen_points(n, point);
    zsort(n*n, point);
    (*data)->point = point;
    (*data)->count = n*n;
    (*data)->beta = beta;
    *kernel = starsh_ssdata_block_exp_kernel;
    return 0;
}
