/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-electrodynamics.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __STARSH_ELECTRODYNAMICS_H__
#define __STARSH_ELECTRODYNAMICS_H__

/*! @defgroup app-electrodynamics Electrodynamics problem
 * @ingroup applications
 * @brief Template for electrodynamics problems.
 *
 * @ref STARSH_eddata holds all the necessary data.
 * */

/*! @defgroup app-electrodynamics-kernels Kernels
 * @ingroup app-electrodynamics
 * @brief Set of kernels for electrodynamics problems
 *
 * Click on functions to view implemented equations.
 * */

// Add definitions for size_t, va_list, STARSH_kernel and STARSH_particles
#include "starsh.h"
#include "starsh-particles.h"

typedef struct starsh_eddata
//! Structure for electrodynamics problems.
/*! @ingroup app-electrodynamics
 * */
{
    STARSH_particles particles;
    //!< Particles.
    //char dtype;
    ////!< Precision of each matrix element (double, single etc).
    double k;
    //!< Characteristical wave number.
    double diag;
    //!< Value of diagonal elements.
} STARSH_eddata;

enum STARSH_ELECTRODYNAMICS_KERNEL
//! List of built-in kernels for starsh_eddata_get_kernel().
/*! For more info on exact formulas inside kernels, take a look at functions
 * starsh_eddata_block_sin_kernel_nd(),
 * starsh_eddata_block_cos_kernel_nd(),
 * starsh_eddata_block_sin_kernel_nd_simd(),
 * starsh_eddata_block_cos_kernel_nd_simd().
 *
 * @ingroup app-electrodynamics
 * */
{
    STARSH_ELECTRODYNAMICS_SIN = 1,
    /*!< Helmholtz sin kernel.
     * @sa starsh_eddata_block_sin_kernel_nd().
     * */
    STARSH_ELECTRODYNAMICS_COS = 2,
    /*!< Helmholtz cos kernel.
     * @sa starsh_eddata_block_cos_kernel_nd().
     * */
    STARSH_ELECTRODYNAMICS_SIN_SIMD = 11,
    /*!< Helmholtz sin SIMD kernel.
     * @sa starsh_eddata_block_sin_kernel_nd_simd().
     * */
    STARSH_ELECTRODYNAMICS_COS_SIMD = 12,
    /*!< Helmholtz cos SIMD kernel.
     * @sa starsh_eddata_block_cos_kernel_nd_simd().
     * */
};

enum STARSH_ELECTRODYNAMICS_PARAM
//! List of parameters for starsh_application().
/*! In the table below each constant corresponds to a given argument and type
 * for starsh_eddata_generate(). These constants are used to generate problem
 * with incomplete set of parameters via starsh_application(),
 * starsh_eddata_generate_va() or starsh_eddata_generate_el().
 *
 * @sa starsh_application(), starsh_eddata_generate(),
 *      starsh_eddata_generate_va(), starsh_eddata_generate_el().
 * @ingroup app-electrodynamics
 * */
{
    STARSH_ELECTRODYNAMICS_NDIM = 1,
    //!< Dimensionality of space (`ndim`, integer).
    STARSH_ELECTRODYNAMICS_K = 2,
    //!< Wave number (`k`, double).
    STARSH_ELECTRODYNAMICS_DIAG = 3,
    //!< Value of diagonal elements (`diag`, double).
    STARSH_ELECTRODYNAMICS_PLACE = 4,
    //!< Distribution of particles (`place`, @ref STARSH_PARTICLES_PLACEMENT).
};

int starsh_eddata_new(STARSH_eddata **data, STARSH_int count, int ndim);
int starsh_eddata_init(STARSH_eddata **data, STARSH_int count, int ndim,
        double *point, double k, double diag);
int starsh_eddata_generate(STARSH_eddata **data, STARSH_int count, int ndim,
        double k, double diag, enum STARSH_PARTICLES_PLACEMENT place);
int starsh_eddata_generate_va(STARSH_eddata **data, STARSH_int count,
        va_list args);
int starsh_eddata_generate_el(STARSH_eddata **data, STARSH_int count, ...);
int starsh_eddata_get_kernel(STARSH_kernel **kernel, STARSH_eddata *data,
         enum STARSH_ELECTRODYNAMICS_KERNEL type);
void starsh_eddata_free(STARSH_eddata *data);

// KERNELS

void starsh_eddata_block_sin_kernel_1d(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);
void starsh_eddata_block_sin_kernel_2d(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);
void starsh_eddata_block_sin_kernel_3d(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);
void starsh_eddata_block_sin_kernel_4d(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);
void starsh_eddata_block_sin_kernel_nd(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);

void starsh_eddata_block_sin_kernel_1d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_eddata_block_sin_kernel_2d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_eddata_block_sin_kernel_3d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_eddata_block_sin_kernel_4d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_eddata_block_sin_kernel_nd_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);

void starsh_eddata_block_cos_kernel_1d(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);
void starsh_eddata_block_cos_kernel_2d(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);
void starsh_eddata_block_cos_kernel_3d(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);
void starsh_eddata_block_cos_kernel_4d(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);
void starsh_eddata_block_cos_kernel_nd(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);

void starsh_eddata_block_cos_kernel_1d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_eddata_block_cos_kernel_2d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_eddata_block_cos_kernel_3d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_eddata_block_cos_kernel_4d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_eddata_block_cos_kernel_nd_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);

#endif // __STARSH_ELECTRODYNAMICS_H__

