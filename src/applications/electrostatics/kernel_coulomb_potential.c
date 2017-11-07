/*! @copyright (c) 2017 King Abdullah University of Science and
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
 * @file src/applications/electrostatics/kernel_coulomb_potential.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 */

#include "common.h"
#include "starsh.h"
#include "starsh-electrostatics.h"

// If dimensionality is static
#if (@NDIM != n)
//! Replace variable ndim with static integer value
#define ndim @NDIM
#endif

void starsh_esdata_block_coulomb_potential_kernel_@NDIMd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld)
//! Coulomb potential for @NDIM-dimensional electrostatics problem.
/*! Fills matrix \f$ A \f$ with values
 * \f[
 *      A_{ij} = \frac{1}{r_{ij}},
 * \f]
 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
 * points. No memory is allocated in this function!
 *
 * @param[in] nrows: Number of rows of \f$ A \f$.
 * @param[in] ncols: Number of columns of \f$ A \f$.
 * @param[in] irow: Array of row indexes.
 * @param[in] icol: Array of column indexes.
 * @param[in] row_data: Pointer to physical data (\ref STARSH_ssdata object).
 * @param[in] col_data: Pointer to physical data (\ref STARSH_ssdata object).
 * @param[out] result: Pointer to memory of \f$ A \f$.
 * @param[in] ld: Leading dimension of `result`.
 * @sa starsh_esdata_block_coulomb_potential_kernel_1d(),
 *      starsh_esdata_block_coulomb_potential_kernel_2d(),
 *      starsh_esdata_block_coulomb_potential_kernel_3d(),
 *      starsh_esdata_block_coulomb_potential_kernel_4d(),
 *      starsh_esdata_block_coulomb_potential_kernel_nd().
 * @ingroup app-electrostatics-kernels
 * */
{
    int i, j, k;
    STARSH_esdata *data1 = row_data;
    STARSH_esdata *data2 = col_data;
    double tmp, dist;
    // Read parameters
// If dimensionality is not static
#if (@NDIM == n)
    int ndim = data1->ndim;
#endif
    // Get coordinates
    STARSH_int count1 = data1->count;
    STARSH_int count2 = data2->count;
    double *x1[ndim], *x2[ndim];
    x1[0] = data1->point;
    x2[0] = data2->point;
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
            if(dist == 0)
                buffer[j*(size_t)ld+i] = 0.0;
            else
                buffer[j*(size_t)ld+i] = 1.0/sqrt(dist);
        }
    }
}

void starsh_esdata_block_coulomb_potential_kernel_@NDIMd_simd(int nrows,
        int ncols, STARSH_int *irow, STARSH_int *icol, void *row_data,
        void *col_data, void *result, int ld)
//! Coulomb potential for @NDIM-dimensional electrostatics problem.
/*! Fills matrix \f$ A \f$ with values
 * \f[
 *      A_{ij} = \frac{1}{r_{ij}},
 * \f]
 * \f$ r_{ij} \f$ is a distance between \f$i\f$-th and \f$j\f$-th spatial
 * points. No memory is allocated in this function!
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
 * @sa starsh_esdata_block_coulomb_potential_kernel_1d_simd(),
 *      starsh_esdata_block_coulomb_potential_kernel_2d_simd(),
 *      starsh_esdata_block_coulomb_potential_kernel_3d_simd(),
 *      starsh_esdata_block_coulomb_potential_kernel_4d_simd(),
 *      starsh_esdata_block_coulomb_potential_kernel_nd_simd().
 * @ingroup app-electrostatics-kernels
 * */
{
    int i, j, k;
    STARSH_esdata *data1 = row_data;
    STARSH_esdata *data2 = col_data;
    double tmp, dist;
    // Read parameters
// If dimensionality is not static
#if (@NDIM == n)
    int ndim = data1->ndim;
#endif
    // Get coordinates
    size_t count1 = data1->count;
    size_t count2 = data2->count;
    double *x1[ndim], *x2[ndim];
    x1[0] = data1->point;
    x2[0] = data2->point;
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
            if(dist == 0)
                buffer[j*(size_t)ld+i] = 0.0;
            else
                buffer[j*(size_t)ld+i] = 1.0/sqrt(dist);
        }
    }
}

