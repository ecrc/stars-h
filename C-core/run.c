#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "misc.h"
#include "cblas.h"
#include "blr.h"
#include "kernel.h"

int main_spatial(int argc, char **argv)
{
    int n = 16, size = n*n; // set random n-by-n particles in 2D
    double tol = 1e-2; // set accuracy tolerance
    srand(time(0));
    double *points = (double *)malloc(2*size*sizeof(double));
    STARS_ssdata data; // Spatial Statistics data
    data.count = size; // number of points
    data.point = points; // pointer to points
    data.beta = .1; // parameter beta fro exponential kernel exp(-r/beta)
    gen_points(n, points); // generate data in array points
    zsort(size, points); // modify array points
    block_func *kernel = STARS_get_kernel("spatial"); // get kernel
    STARS_blr_compress_uniform(n, n, &data, kernel, tol); // get compressed
        // matrix
    return 0;
}

int main(int argc, char **argv)
{
    int bsize = 16; // set number of rows/columns of a block of a matrix
    int bcount = 16; // set number of row/column blocks of a matrix
    double tol = 1e-2; // set accuracy tolerance
    srand(time(0));
    STARS_synthdata data; // Synthetic data
    data.count = bsize*bcount; // number of points
    data.bcount = bcount;
    data.bsize = bsize;
    data.U = (double *)malloc(bsize*bsize*bcount*sizeof(double));
    block_func *kernel = STARS_get_kernel("synth"); // get kernel
    STARS_blr_compress_uniform(bcount, bsize, &data, kernel, tol); // get compressed
        // matrix
    return 0;
}
