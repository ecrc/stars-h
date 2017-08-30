/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/applications/spatial.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-22
 */

#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"

void starsh_ssdata_block_exp_kernel_1d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Exponential kernel for spatial statistics problem in 1D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns exp^{-r/beta}, where r is a distance between particles in 1D
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -data->beta;
    double noise = data->noise;
    // Get X coordinate
    double *x = data->particles.point;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = fabs(x[irow[i]]-x[icol[j]]);
            dist = tmp/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_exp_kernel_1d(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result)
//! Exponential kernel for spatial statistics problem in 1D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns exp^{-r/beta}, where r is a distance between particles in 1D
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -data->beta;
    double noise = data->noise;
    // Get X coordinate
    double *x = data->particles.point;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = fabs(x[irow[i]]-x[icol[j]]);
            dist = tmp/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_exp_kernel_2d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Exponential kernel for spatial statistics problem in 2D
/*! Block kernel with SIMD instructions for spatial statistics
 * Returns exp^{-r/beta}, where r is a distance between particles in 2D
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -data->beta;
    double noise = data->noise;
    // Get X and Y coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_exp_kernel_2d(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result)
//! Exponential kernel for spatial statistics problem in 2D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns exp^{-r/beta}, where r is a distance between particles in 2D
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -data->beta;
    double noise = data->noise;
    // Get X and Y coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_exp_kernel_3d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Exponential kernel for spatial statistics problem in 3D
/*! Block kernel with SIMD instructions for spatial statistics
 * Returns exp^{-r/beta}, where r is a distance between particles in 3D
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -data->beta;
    double noise = data->noise;
    // Get X, Y and Z coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count, *z = y+count;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            tmp = z[irow[i]]-z[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_exp_kernel_3d(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result)
//! Exponential kernel for spatial statistics problem in 3D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns exp^{-r/beta}, where r is a distance between particles in 3D
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -data->beta;
    double noise = data->noise;
    // Get X, Y and Z coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count, *z = y+count;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            tmp = z[irow[i]]-z[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_sqr_exp_kernel_1d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Square exponential kernel for spatial statistics problem in 1D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns exp^{-r^2/(2 beta^2)}, where r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -2*data->beta*data->beta;
    double noise = data->noise;
    // Get X coordinates
    double *x = data->particles.point;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            dist = dist/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_sqr_exp_kernel_1d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Square exponential kernel for spatial statistics problem in 1D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns exp^{-r^2/(2 beta^2)}, where r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -2*data->beta*data->beta;
    double noise = data->noise;
    // Get X coordinates
    double *x = data->particles.point;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            dist = dist/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_sqr_exp_kernel_2d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Square exponential kernel for spatial statistics problem in 2D
/*! Block kernel for spatial statistics
 * Returns exp^{-r^2/(2 beta^2)}, where r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -2*data->beta*data->beta;
    double noise = data->noise;
    // Get X and Y coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = dist/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_sqr_exp_kernel_2d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Square exponential kernel for spatial statistics problem in 2D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns exp^{-r^2/(2 beta^2)}, where r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -2*data->beta*data->beta;
    double noise = data->noise;
    // Get X and Y coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = dist/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_sqr_exp_kernel_3d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Square exponential kernel for spatial statistics problem in 3D
/*! Block kernel with SIMD instructions for spatial statistics
 * Returns exp^{-r^2/(2 beta^2)}, where r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -2*data->beta*data->beta;
    double noise = data->noise;
    // Get X, Y and Z coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count, *z = y+count;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            tmp = z[irow[i]]-z[icol[j]];
            dist += tmp*tmp;
            dist = dist/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

void starsh_ssdata_block_sqr_exp_kernel_3d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Square exponential kernel for spatial statistics problem in 3D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns exp^{-r^2/(2 beta^2)}, where r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta and noise
    double tmp, dist, beta = -2*data->beta*data->beta;
    double noise = data->noise;
    // Get X, Y and Z coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count, *z = y+count;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            tmp = z[irow[i]]-z[icol[j]];
            dist += tmp*tmp;
            dist = dist/beta;
            if(dist == 0)
                buffer[j*(size_t)nrows+i] = 1+noise;
            else
                buffer[j*(size_t)nrows+i] = exp(dist);
        }
}

#ifdef GSL
void starsh_ssdata_block_matern_kernel_1d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Matern kernel for spatial statistics problem in 1D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=sqrt(2*nu)*r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    double theta = sqrt(2*nu)/beta;
    // Get X coordinates
    double *x = data->particles.point;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = abs(tmp);
            dist = dist*theta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern_kernel_1d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Matern kernel for spatial statistics problem in 1D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=sqrt(2*nu)*r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    double theta = sqrt(2*nu)/beta;
    // Get X coordinates
    double *x = data->particles.point;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = abs(tmp);
            dist = dist*theta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern_kernel_2d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Matern kernel for spatial statistics problem in 2D
/*! Block kernel with SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=sqrt(2*nu)*r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    double theta = sqrt(2*nu)/beta;
    // Get X and Y coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)*theta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern_kernel_2d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Matern kernel for spatial statistics problem in 2D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=sqrt(2*nu)*r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    double theta = sqrt(2*nu)/beta;
    // Get X and Y coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)*theta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern_kernel_3d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Matern kernel for spatial statistics problem in 3D
/*! Block kernel with SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=sqrt(2*nu)*r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    double theta = sqrt(2*nu)/beta;
    // Get X, Y and Z coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count, *z = y+count;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            tmp = z[irow[i]]-z[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)*theta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern_kernel_3d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Matern kernel for spatial statistics problem in 3D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=sqrt(2*nu)*r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    double theta = sqrt(2*nu)/beta;
    // Get X, Y and Z coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count, *z = y+count;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            tmp = z[irow[i]]-z[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)*theta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern2_kernel_1d_simd(int nrows,
        int ncols, int *irow, int *icol, void *row_data, void *col_data,
        void *result)
//! Matern kernel for spatial statistics problem in 1D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    // Get X coordinates
    double *x = data->particles.point;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = abs(tmp);
            dist = dist/beta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern2_kernel_1d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Matern kernel for spatial statistics problem in 1D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    // Get X coordinates
    double *x = data->particles.point;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = abs(tmp);
            dist = dist/beta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern2_kernel_2d_simd(int nrows,
        int ncols, int *irow, int *icol, void *row_data, void *col_data,
        void *result)
//! Matern kernel for spatial statistics problem in 2D
/*! Block kernel with SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    // Get X and Y coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern2_kernel_2d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Matern kernel for spatial statistics problem in 2D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    // Get X and Y coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern2_kernel_3d_simd(int nrows,
        int ncols, int *irow, int *icol, void *row_data, void *col_data,
        void *result)
//! Matern kernel for spatial statistics problem in 3D
/*! Block kernel with SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    // Get X, Y and Z coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count, *z = y+count;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            tmp = z[irow[i]]-z[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}

void starsh_ssdata_block_matern2_kernel_3d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result)
//! Matern kernel for spatial statistics problem in 3D
/*! Block kernel without SIMD instructions for spatial statistics
 * Returns 2^(1-nu)/Gamma(nu)*x^nu*K_nu(x), where x=r/beta
 * and r is a distance between particles
 * @ingroup applications
 * @param[in] nrows: Number of rows of corresponding array.
 * @param[in] ncols: Number of columns of corresponding array.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data.
 * @param[in] col_data: Pointer to physical data.
 * @param[out] result: Where to write elements of an array.
 * */
{
    int i, j;
    STARSH_ssdata *data = row_data;
    // Read parameters beta, nu and noise
    double tmp, dist, beta = data->beta, nu = data->nu;
    double noise = data->noise;
    // Get X, Y and Z coordinates
    size_t count = data->particles.count;
    double *x = data->particles.point, *y = x+count, *z = y+count;
    double *buffer = result;
    // Fill column-major matrix
    for(j = 0; j < ncols; j++)
        for(i = 0; i < nrows; i++)
        {
            tmp = x[irow[i]]-x[icol[j]];
            dist = tmp*tmp;
            tmp = y[irow[i]]-y[icol[j]];
            dist += tmp*tmp;
            tmp = z[irow[i]]-z[icol[j]];
            dist += tmp*tmp;
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*nrows+i] = 1.0+noise;
            else
                buffer[j*nrows+i] = pow(2.0, 1-nu)/gsl_sf_gamma(nu)*
                    pow(dist, nu)*gsl_sf_bessel_Knu(nu, dist);
        }
}
#endif


int starsh_ssdata_new(STARSH_ssdata **data, int n, char dtype, int ndim,
        double beta, double nu, double noise, int place)
//! Generate spatial statistics data.
/*! @ingroup applications
 * @param[out] data: Address of pointer to `STARSH_ssdata` object.
 * @param[in] n: Size of matrix.
 * @param[in] dtype: Precision ('s', 'd', 'c' or 'z').
 * @param[in] ndim: Dimensionality of spatial statisics problem.
 * @param[in] beta: Parameter for kernel.
 * @param[in] nu: Smoothing parameter for Matern kernel.
 * @param[in] noise: Value to add to diagonal elements.
 * @param[in] place: 1 for generating rectangular grid of particles with
 *      random shifts to grid lines, 0 for generating each particle randomly.
 * @return Error code.
 * */
{
    if(data == NULL)
    {
        STARSH_ERROR("invalid value of `data`");
        return 1;
    }
    if(n <= 0)
    {
        STARSH_ERROR("invalid value of `n`");
        return 1;
    }
    if(beta <= 0)
    {
        STARSH_ERROR("invalid value of `beta`");
        return 1;
    }
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return 1;
    }
    if(noise < 0)
    {
        STARSH_ERROR("invalid value of `noise`");
        return 1;
    }
    int info;
    STARSH_particles *particles;
    info = starsh_particles_generate(&particles, n, ndim, place);
    if(info != 0)
    {
        fprintf(stderr, "INFO=%d\n", info);
        return info;
    }
    STARSH_MALLOC(*data, 1);
    (*data)->particles = *particles;
    free(particles);
    (*data)->dtype = dtype;
    (*data)->beta = beta;
    (*data)->nu = nu;
    (*data)->noise = noise;
    return 0;
}

int starsh_ssdata_new_va(STARSH_ssdata **data, int n, char dtype,
        va_list args)
//! Generate spatial statistics data with va_list.
//! For more info look at starsh_ssdata_new().
//! @ingroup applications
{
    int arg_type;
    // Set default values
    int ndim = 2;
    double beta = 0.1;
    double nu = 0.5;
    double noise = 0;
    int place = STARSH_PARTICLES_UNIFORM;
    int info;
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return 1;
    }
    while((arg_type = va_arg(args, int)) != 0)
    {
        switch(arg_type)
        {
            case STARSH_SPATIAL_NDIM:
                ndim = va_arg(args, int);
                break;
            case STARSH_SPATIAL_BETA:
                beta = va_arg(args, double);
                break;
            case STARSH_SPATIAL_NU:
                nu = va_arg(args, double);
                break;
            case STARSH_SPATIAL_NOISE:
                noise = va_arg(args, double);
                break;
            case STARSH_SPATIAL_PLACE:
                place = va_arg(args, int);
                break;
            default:
                STARSH_ERROR("Wrong parameter type");
                return STARSH_WRONG_PARAMETER;
        }
    }
    info = starsh_ssdata_new(data, n, dtype, ndim, beta, nu, noise, place);
    return info;
}

int starsh_ssdata_new_el(STARSH_ssdata **data, int n, char dtype, ...)
//! Generate spatial statistics data with ellipsis.
//! For more info look at starsh_ssdata_new().
//! @ingroup applications
{
    va_list args;
    va_start(args, dtype);
    int info = starsh_ssdata_new_va(data, n, dtype, args);
    va_end(args);
    return info;
}

void starsh_ssdata_free(STARSH_ssdata *data)
//! Free data.
//! @ingroup applications
{
    starsh_particles_free(&data->particles);
}

static int starsh_ssdata_get_kernel_1d(STARSH_kernel *kernel, int type,
        char dtype)
//! Get corresponding kernel for 1D spatial statistics problem.
//! @ingroup applications
{
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return 1;
    }
    switch(type)
    {
        case STARSH_SPATIAL_EXP:
            *kernel = starsh_ssdata_block_exp_kernel_1d;
            break;
        case STARSH_SPATIAL_EXP_SIMD:
            *kernel = starsh_ssdata_block_exp_kernel_1d_simd;
            break;
        case STARSH_SPATIAL_SQREXP:
            *kernel = starsh_ssdata_block_sqr_exp_kernel_1d;
            break;
        case STARSH_SPATIAL_SQREXP_SIMD:
            *kernel = starsh_ssdata_block_sqr_exp_kernel_1d_simd;
            break;
#ifdef GSL
        case STARSH_SPATIAL_MATERN:
            *kernel = starsh_ssdata_block_matern_kernel_1d;
            break;
        case STARSH_SPATIAL_MATERN_SIMD:
            *kernel = starsh_ssdata_block_matern_kernel_1d_simd;
            break;
        case STARSH_SPATIAL_MATERN2:
            *kernel = starsh_ssdata_block_matern2_kernel_1d;
            break;
        case STARSH_SPATIAL_MATERN2_SIMD:
            *kernel = starsh_ssdata_block_matern2_kernel_1d_simd;
            break;
#endif
        default:
            STARSH_ERROR("Wrong type of kernel");
            return 1;
    }
    return 0;
}

static int starsh_ssdata_get_kernel_2d(STARSH_kernel *kernel, int type,
        char dtype)
//! Get corresponding kernel for 2D spatial statistics problem.
//! @ingroup applications
{
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return 1;
    }
    switch(type)
    {
        case STARSH_SPATIAL_EXP:
            *kernel = starsh_ssdata_block_exp_kernel_2d;
            break;
        case STARSH_SPATIAL_EXP_SIMD:
            *kernel = starsh_ssdata_block_exp_kernel_2d_simd;
            break;
        case STARSH_SPATIAL_SQREXP:
            *kernel = starsh_ssdata_block_sqr_exp_kernel_2d;
            break;
        case STARSH_SPATIAL_SQREXP_SIMD:
            *kernel = starsh_ssdata_block_sqr_exp_kernel_2d_simd;
            break;
#ifdef GSL
        case STARSH_SPATIAL_MATERN:
            *kernel = starsh_ssdata_block_matern_kernel_2d;
            break;
        case STARSH_SPATIAL_MATERN_SIMD:
            *kernel = starsh_ssdata_block_matern_kernel_2d_simd;
            break;
        case STARSH_SPATIAL_MATERN2:
            *kernel = starsh_ssdata_block_matern2_kernel_2d;
            break;
        case STARSH_SPATIAL_MATERN2_SIMD:
            *kernel = starsh_ssdata_block_matern2_kernel_2d_simd;
            break;
#endif
        default:
            STARSH_ERROR("Wrong type of kernel");
            return 1;
    }
    return 0;
}

static int starsh_ssdata_get_kernel_3d(STARSH_kernel *kernel, int type,
        char dtype)
//! Get corresponding kernel for 3D spatial statistics problem.
//! @ingroup applications
{
    if(dtype != 'd')
    {
        STARSH_ERROR("Only dtype='d' is supported");
        return 1;
    }
    switch(type)
    {
        case STARSH_SPATIAL_EXP:
            *kernel = starsh_ssdata_block_exp_kernel_3d;
            break;
        case STARSH_SPATIAL_EXP_SIMD:
            *kernel = starsh_ssdata_block_exp_kernel_3d_simd;
            break;
        case STARSH_SPATIAL_SQREXP:
            *kernel = starsh_ssdata_block_sqr_exp_kernel_3d;
            break;
        case STARSH_SPATIAL_SQREXP_SIMD:
            *kernel = starsh_ssdata_block_sqr_exp_kernel_3d_simd;
            break;
#ifdef GSL
        case STARSH_SPATIAL_MATERN:
            *kernel = starsh_ssdata_block_matern_kernel_3d;
            break;
        case STARSH_SPATIAL_MATERN_SIMD:
            *kernel = starsh_ssdata_block_matern_kernel_3d_simd;
            break;
        case STARSH_SPATIAL_MATERN2:
            *kernel = starsh_ssdata_block_matern2_kernel_3d;
            break;
        case STARSH_SPATIAL_MATERN2_SIMD:
            *kernel = starsh_ssdata_block_matern2_kernel_3d_simd;
            break;
#endif
        default:
            STARSH_ERROR("Wrong type of kernel");
            return 1;
    }
    return 0;
}

int starsh_ssdata_get_kernel(STARSH_kernel *kernel, STARSH_ssdata *data,
        int type)
//! Get corresponding kernel for spatial statistics problem.
//! @ingroup applications
{
    switch(data->particles.ndim)
    {
        case 1:
            return starsh_ssdata_get_kernel_1d(kernel, type, data->dtype);
        case 2:
            return starsh_ssdata_get_kernel_2d(kernel, type, data->dtype);
        case 3:
            return starsh_ssdata_get_kernel_3d(kernel, type, data->dtype);
        default:
            STARSH_ERROR("parameter ndim=%d is not supported (must be 1, 2 or "
                    "3)", data->particles.ndim);
            return 1;
    }
}
