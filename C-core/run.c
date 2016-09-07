#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "misc.h"
#include "cblas.h"
#include "block_lowrank.h"

typedef struct
{
    int count;
    double *point;
} my_data;

double exp_kernel(double x)
{
    return exp(-x);
}

void block_kernel(int n, double *points, int bi, int bj, int m,
        double kernel(double), double *A)
{
    int i, j;
    double *B = points+n;
    double tmp, dist;
    // Column cycle (i is a number of column, not a number of row)
    for(i = 0; i < m; i++)
    {
        // For each element of i-th column
        for(j = 0; j < m; j++)
        {
            tmp = points[bi+j]-points[bj+i];
            dist = tmp*tmp;
            tmp = B[bi+j]-B[bj+i];
            dist += tmp*tmp;
            dist = sqrt(dist);
            A[i*m+j] = kernel(dist);
        }
    }
}

void block_exp_kernel(int rows, int cols, int *row, int *col,
        my_data *row_data, my_data *col_data, double *result)
{
    int i, j;
    double tmp, dist;
    double *x = row_data->point;
    double *y = row_data->point+row_data->count;
    for(j = 0; j < cols; j++)
        for(i = 0; i < rows; i++)
        {
            tmp = x[row[i]]-x[col[j]];
            dist = tmp*tmp;
            tmp = y[row[i]]-y[col[j]];
            dist += tmp*tmp;
            dist = sqrt(dist);
            result[j*rows+i] = exp_kernel(dist);
        }
}

int main(int argc, char **argv)
{
    int n = 16, size = n*n;
    double tol = 1e-2;
    srand(time(0));
    double *points = (double *)malloc(2*size*sizeof(double));
    my_data data;
    data.count = size;
    data.point = points;
    gen_points(n, points);
    unified_compress_symm(n, n, &data, block_exp_kernel, tol);
    return 0;
}
