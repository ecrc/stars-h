#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "blr.h"
#include "kernel.h"

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
    else
    {
        printf("Wrong parameter to STARS_get_kernel\n");
        return NULL;
    }
}
