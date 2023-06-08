/*! @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @generate NDIM -> n 1 2 3 4
 * Generate different functions for different dimensions. This hack improves
 * performance in certain cases. Value 'n' stands for general case, whereas all
 * other values correspond to static values of dimensionality.
 * During code generation step, each appearance of @NDIM (including this one)
 * will be replace by proposed values. If you want to use this file outside
 * STARS-H, simply do substitutions yourself.
 *
 * @file src/applications/spatial/kernel_matern2.c
 * @version 0.3.1
 * @author Aleksandr Mikhalev
 * @date 2018-11-06
 */

#include "common.h"
#include "starsh.h"
#include "starsh-spatial.h"

// If dimensionality is static
#if (@NDIM != n)
//! Replace variable ndim with static integer value
#define ndim @NDIM
#endif

#ifdef GSL

void starsh_ssdata_block_matern2_kernel_@NDIMd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld)
//! Mat&eacute;rn kernel for @NDIM-dimensional spatial statistics problem
/*! Fills matrix \f$ A \f$ with values
 * \f[
 *      A_{ij} = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{r_{ij}}
 *      {\beta} \right)^{\nu} K_{\nu} \left( \frac{r_{ij}}{\beta} \right) +
 *      \mu \delta(r_{ij}),
 * \f]
 * where \f$ \Gamma \f$ is the Gamma function, \f$ K_{\nu} \f$ is the modified
 * Bessel function of the second kind, \f$ \delta \f$ is the delta function
 * \f[
 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
 *      \end{array} \right.,
 * \f]
 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
 * points and variance \f$ \sigma \f$, correlation length \f$ \beta \f$,
 * smoothing parameter \f$ \nu \f$ and noise \f$ \mu \f$ come from \p
 * row_data (\ref STARSH_ssdata object). No memory is allocated in this
 * function!
 *
 * @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * @sa starsh_ssdata_block_matern2_kernel_1d(),
 *      starsh_ssdata_block_matern2_kernel_2d(),
 *      starsh_ssdata_block_matern2_kernel_3d(),
 *      starsh_ssdata_block_matern2_kernel_4d(),
 *      starsh_ssdata_block_matern2_kernel_nd().
 * @ingroup app-spatial-kernels
 * */
{
    int i, j, k;
    STARSH_ssdata *data1 = row_data;
    STARSH_ssdata *data2 = col_data;
    double tmp, dist;
    // Read parameters
// If dimensionality is not static
#if (@NDIM == n)
    int ndim = data1->particles.ndim;
#endif
    double beta = data1->beta;
    double nu = data1->nu;
    double noise = data1->noise;
    double sigma = data1->sigma;
    // Get coordinates
    STARSH_int count1 = data1->particles.count;
    STARSH_int count2 = data2->particles.count;
    double *x1[ndim], *x2[ndim];
    x1[0] = data1->particles.point;
    x2[0] = data2->particles.point;
    //#pragma omp simd
    for(i = 1; i < ndim; i++)
    {
        x1[i] = x1[0]+i*count1;
        x2[i] = x2[0]+i*count2;
    }
    double *x1_cur, *x2_cur;
    double *buffer = result;
    // Fill column-major matrix
    //#pragma omp simd
    for(j = 0; j < ncols; j++)
    {
        for(i = 0; i < nrows; i++)
        {
            dist = 0.0;
            for(k = 0; k < ndim; k++)
            {
                tmp = x1[k][irow[i]]-x2[k][icol[j]];
                dist += tmp*tmp;
            }
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*(size_t)ld+i] = sigma+noise;
            else
                buffer[j*(size_t)ld+i] = sigma*pow(2.0, 1.0-nu)/
                        gsl_sf_gamma(nu)*pow(dist, nu)*
                        gsl_sf_bessel_Knu(nu, dist);
        }
    }
}

void starsh_ssdata_block_matern2_kernel_@NDIMd_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld)
//! Mat&eacute;rn kernel for @NDIM-dimensional spatial statistics problem
/*! Fills matrix \f$ A \f$ with values
 * \f[
 *      A_{ij} = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{r_{ij}}
 *      {\beta} \right)^{\nu} K_{\nu} \left( \frac{r_{ij}}{\beta} \right) +
 *      \mu \delta(r_{ij}),
 * \f]
 * where \f$ \Gamma \f$ is the Gamma function, \f$ K_{\nu} \f$ is the modified
 * Bessel function of the second kind, \f$ \delta \f$ is the delta function
 * \f[
 *      \delta(x) = \left\{ \begin{array}{ll} 0, & x \ne 0\\ 1, & x = 0
 *      \end{array} \right.,
 * \f]
 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
 * points and variance \f$ \sigma \f$, correlation length \f$ \beta \f$,
 * smoothing parameter \f$ \nu \f$ and noise \f$ \mu \f$ come from \p
 * row_data (\ref STARSH_ssdata object). No memory is allocated in this
 * function!
 *
 * Uses SIMD instructions.
 *
 * @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * @sa starsh_ssdata_block_matern2_kernel_1d_simd(),
 *      starsh_ssdata_block_matern2_kernel_2d_simd(),
 *      starsh_ssdata_block_matern2_kernel_3d_simd(),
 *      starsh_ssdata_block_matern2_kernel_4d_simd(),
 *      starsh_ssdata_block_matern2_kernel_nd_simd().
 * @ingroup app-spatial-kernels
 * */
{
    int i, j, k;
    STARSH_ssdata *data1 = row_data;
    STARSH_ssdata *data2 = col_data;
    double tmp, dist;
    // Read parameters
// If dimensionality is not static
#if (@NDIM == n)
    int ndim = data1->particles.ndim;
#endif
    double beta = data1->beta;
    double nu = data1->nu;
    double noise = data1->noise;
    double sigma = data1->sigma;
    // Get coordinates
    STARSH_int count1 = data1->particles.count;
    STARSH_int count2 = data2->particles.count;
    double *x1[ndim], *x2[ndim];
    x1[0] = data1->particles.point;
    x2[0] = data2->particles.point;
    #pragma omp simd
    for(i = 1; i < ndim; i++)
    {
        x1[i] = x1[0]+i*count1;
        x2[i] = x2[0]+i*count2;
    }
    double *x1_cur, *x2_cur;
    double *buffer = result;
    // Fill column-major matrix
    #pragma omp simd
    for(j = 0; j < ncols; j++)
    {
        for(i = 0; i < nrows; i++)
        {
            dist = 0.0;
            for(k = 0; k < ndim; k++)
            {
                tmp = x1[k][irow[i]]-x2[k][icol[j]];
                dist += tmp*tmp;
            }
            dist = sqrt(dist)/beta;
            if(dist == 0)
                buffer[j*(size_t)ld+i] = sigma+noise;
            else
                buffer[j*(size_t)ld+i] = sigma*pow(2.0, 1.0-nu)/
                        gsl_sf_gamma(nu)*pow(dist, nu)*
                        gsl_sf_bessel_Knu(nu, dist);
        }
    }
}

#endif // GSL

