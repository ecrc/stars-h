/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * @file starsh-spatial.h
 * @version 1.0.0.2
 * @author Aleksandr Mikhalev
 * @date 16 May 2017
 * */

#ifndef __STARSH_SPATIAL_H__
#define __STARSH_SPATIAL_H__

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"

typedef struct starsh_ssdata
//! Structure for Spatial Statistics problems.
{
    size_t count;
    //!< Number of spatial points.
    int ndim;
    //!< Dimensionality of the problem.
    char dtype;
    //!< Precision of each matrix element (double, single etc)
    double *point;
    //!< Coordinates of spatial points.
    double beta;
    //!< Characteristical length of covariance. Notes as l in some papers.
    double nu;
    //!< Order of Matern kernel.
    double noise;
    //!< Noise and regularization parameter.
} STARSH_ssdata;

enum STARSH_SPATIAL_KERNEL
{
    STARSH_SPATIAL_EXP = 1,
    STARSH_SPATIAL_SQREXP = 2,
    STARSH_SPATIAL_MATERN = 3,
    STARSH_SPATIAL_MATERN2 = 4,
    STARSH_SPATIAL_EXP_SIMD = 11,
    STARSH_SPATIAL_SQREXP_SIMD = 12,
    STARSH_SPATIAL_MATERN_SIMD = 13,
    STARSH_SPATIAL_MATERN2_SIMD = 14
};

enum STARSH_SPATIAL_PARAM
{
    STARSH_SPATIAL_NDIM = 1,
    STARSH_SPATIAL_BETA = 2,
    STARSH_SPATIAL_NU = 3,
    STARSH_SPATIAL_NOISE = 4,
    STARSH_SPATIAL_PLACE = 5
};

int starsh_ssdata_new(STARSH_ssdata **data, int n, char dtype, int ndim,
        double beta, double nu, double noise, int place);
int starsh_ssdata_new_va(STARSH_ssdata **data, const int n, char dtype,
        va_list args);
int starsh_ssdata_new_el(STARSH_ssdata **data, const int n, char dtype, ...);
int starsh_ssdata_get_kernel(STARSH_kernel *kernel, STARSH_ssdata *data,
        int type);
void starsh_ssdata_free(STARSH_ssdata *data);

// KERNELS
void starsh_ssdata_block_exp_kernel_1d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_exp_kernel_1d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_exp_kernel_2d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_exp_kernel_2d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_exp_kernel_3d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_exp_kernel_3d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);

void starsh_ssdata_block_sqrexp_kernel_1d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_sqrexp_kernel_1d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_sqrexp_kernel_2d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_sqrexp_kernel_2d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_sqrexp_kernel_3d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_sqrexp_kernel_3d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);

void starsh_ssdata_block_matern_kernel_1d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern_kernel_1d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern_kernel_2d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern_kernel_2d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern_kernel_3d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern_kernel_3d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);

void starsh_ssdata_block_matern2_kernel_1d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern2_kernel_1d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern2_kernel_2d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern2_kernel_2d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern2_kernel_3d_simd(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);
void starsh_ssdata_block_matern2_kernel_3d(int nrows, int ncols,
        int *irow, int *icol, void *row_data, void *col_data, void *result);

#endif // __STARSH_SPATIAL_H__
