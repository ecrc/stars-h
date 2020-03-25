/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-spatial.h
 * @version 0.1.1
 * @author Aleksandr Mikhalev
 * @date 2018-11-06
 * */

#ifndef __STARSH_SPATIAL_H__
#define __STARSH_SPATIAL_H__

/*! @defgroup app-spatial Spatial statistics problem
 * @ingroup applications
 * @brief Template for spatial statistics problems.
 *
 * @ref STARSH_ssdata holds all the necessary data.
 * */

/*! @defgroup app-spatial-kernels Kernels
 * @ingroup app-spatial
 * @brief Set of kernels for spatial statistics problems
 *
 * Click on functions to view implemented equations.
 * */

// Add definitions for size_t, va_list, STARSH_kernel and STARSH_particles
#include "starsh.h"
#include "starsh-particles.h"

typedef struct starsh_ssdata
//! Structure for Spatial Statistics problems.
/*! @ingroup app-spatial
 * */
{
    STARSH_particles particles;
    //!< Particles.
    //char dtype;
    ////!< Precision of each matrix element (double, single etc).
    double beta;
    //!< Characteristical length of covariance.
    double nu;
    //!< Order of Mat&eacute;rn kernel.
    double noise;
    //!< Noise and regularization parameter.
    double sigma;
    //!< Variance.
    double nu2;
    //!< Order of Mat&eacute;rn kernel.
    double noise2;
    //!< Noise and regularization parameter.
    double sigma2;
    //!< Variance.
   double corr;
} STARSH_ssdata;

enum STARSH_SPATIAL_KERNEL
//! List of built-in kernels for starsh_ssdata_get_kernel().
/*! For more info on exact formulas inside kernels, take a look at functions
 * starsh_ssdata_block_exp_kernel_nd(),
 * starsh_ssdata_block_sqrexp_kernel_nd(),
 * starsh_ssdata_block_matern_kernel_nd(),
 * starsh_ssdata_block_matern2_kernel_nd(),
 * starsh_ssdata_block_exp_kernel_nd_simd(),
 * starsh_ssdata_block_sqrexp_kernel_nd_simd(),
 * starsh_ssdata_block_matern_kernel_nd_simd(),
 * starsh_ssdata_block_matern2_kernel_nd_simd().
 *
 * @ingroup app-spatial
 * */
{
    STARSH_SPATIAL_EXP = 1,
    /*!< Exponential kernel.
     * @sa starsh_ssdata_block_exp_kernel_nd().
     * */
    STARSH_SPATIAL_SQREXP = 2,
    /*!< Square exponential kernel.
     * @sa starsh_ssdata_block_sqrexp_kernel_nd().
     * */
    STARSH_SPATIAL_MATERN = 3,
    /*!< Mat&eacute;rn kernel.
     * @sa starsh_ssdata_block_matern_kernel_nd().
     * */
    STARSH_SPATIAL_MATERN2 = 4,
    /*!< Modified Mat&eacute;rn kernel.
     * @sa starsh_ssdata_block_matern2_kernel_nd().
     * */
    STARSH_SPATIAL_EXP_SIMD = 11,
    /*!< Exponential kernel with SIMD.
     * @sa starsh_ssdata_block_exp_kernel_nd_simd().
     * */
    STARSH_SPATIAL_SQREXP_SIMD = 12,
    /*!< Exponential kernel with SIMD.
     * @sa starsh_ssdata_block_sqrexp_kernel_nd_simd().
     * */
    STARSH_SPATIAL_MATERN_SIMD = 13,
    /*!< Mat&eacute;rn kernel with SIMD.
     * @sa starsh_ssdata_block_matern_kernel_nd().
     * */
    STARSH_SPATIAL_MATERN2_SIMD = 14,
    /*!< Modified Mat&eacute;rn kernel with SIMD.
     * @sa starsh_ssdata_block_matern2_kernel_nd_simd().
     * */
    STARSH_SPATIAL_EXP_GCD = 15,
    /*!< Exponential kernel.
     * @sa starsh_ssdata_block_exp_kernel_nd_simd_gcd().
     * */
    STARSH_SPATIAL_SQREXP_GCD = 16,
    /*!< Square exponential kernel.
     * @sa starsh_ssdata_block_sqrexp_kernel_nd_simd_gcd().
     * */
    STARSH_SPATIAL_MATERN_GCD = 17,
    /*!< Mat&eacute;rn kernel with SIMD.
     * @sa starsh_ssdata_block_matern_kernel_nd_simd_gcd().
     * */
    STARSH_SPATIAL_MATERN2_GCD = 18,
    /*!< Modified Mat&eacute;rn kernel with SIMD.
     * @sa starsh_ssdata_block_matern2_kernel_nd_simd_gcd().
     * */
    STARSH_SPATIAL_PARSIMONIOUS_GCD = 19,
    /*!< Bivariate Modified parsimonious kernel.
     * @sa starsh_ssdata_block_parsimonious_kernel_nd_simd_gcd().
     * */
    STARSH_SPATIAL_PARSIMONIOUS2_GCD = 20,
    /*!< Bivariate Modified parsimonious2 kernel.
     * @sa starsh_ssdata_block_parsimonious_kernel_nd_simd_gcd().
     * */
    STARSH_SPATIAL_PARSIMONIOUS_SIMD = 21,
    /*!< Bivariate Modified parsimonious kernel.
     * @sa starsh_ssdata_block_parsimonious_kernel_nd_simd().
     * */
    STARSH_SPATIAL_PARSIMONIOUS2_SIMD = 22,
    /*!< Bivariate Modified parsimonious2 kernel.
     * @sa starsh_ssdata_block_parsimonious_kernel_nd_simd().
     * */
};

enum STARSH_SPATIAL_PARAM
//! List of parameters for starsh_application().
/*! In the table below each constant corresponds to a given argument and type
 * for starsh_ssdata_generate(). These constants are used to generate problem
 * with incomplete set of parameters via starsh_application(),
 * starsh_ssdata_generate_va() or starsh_ssdata_generate_el().
 *
 * @sa starsh_application(), starsh_ssdata_generate(),
 *      starsh_ssdata_generate_va(), starsh_ssdata_generate_el().
 * @ingroup app-spatial
 * */
{
    STARSH_SPATIAL_NDIM = 1,
    //!< Dimensionality of space (`ndim`, integer).
    STARSH_SPATIAL_BETA = 2,
    //!< Correlation length (`beta`, double).
    STARSH_SPATIAL_NU = 3,
    //!< Smoothing parameter for Mat&eacute;rn kernel (`nu`, double).
    STARSH_SPATIAL_NOISE = 4,
    //!< Noise or what to add to diagonal elements (`noise`, double).
    STARSH_SPATIAL_PLACE = 5,
    //!< Distribution of particles (`place`, @ref STARSH_PARTICLES_PLACEMENT).
    STARSH_SPATIAL_SIGMA = 6,
    //!< Variance parameter (`sigma`, double).
    STARSH_SPATIAL_SIGMA2 = 7,
    //!< Variance parameter (`sigma`, double).
    STARSH_SPATIAL_NU2 = 8,
    //!< Smoothing parameter for Mat&eacute;rn kernel (`nu`, double).
    STARSH_SPATIAL_CORR = 9,
};

int starsh_ssdata_new(STARSH_ssdata **data, STARSH_int count, int ndim);
int starsh_ssdata_init(STARSH_ssdata **data, STARSH_int count, int ndim,
		double *point, double beta, double nu, double noise, double sigma);
int starsh_ssdata_generate(STARSH_ssdata **data, STARSH_int count, int ndim,
		double beta, double nu, double noise,
		enum STARSH_PARTICLES_PLACEMENT place, double sigma);
int starsh_ssdata_generate_va(STARSH_ssdata **data, STARSH_int count,
		va_list args);
int starsh_ssdata_generate_el(STARSH_ssdata **data, STARSH_int count, ...);
int starsh_ssdata_get_kernel(STARSH_kernel **kernel, STARSH_ssdata *data,
		enum STARSH_SPATIAL_KERNEL type);
void starsh_ssdata_free(STARSH_ssdata *data);

// KERNELS

void starsh_ssdata_block_exp_kernel_1d(int nrows, int ncols, STARSH_int *irow,
		STARSH_int *icol, void *row_data, void *col_data, void *result,
		int ld);
void starsh_ssdata_block_exp_kernel_2d(int nrows, int ncols, STARSH_int *irow,
		STARSH_int *icol, void *row_data, void *col_data, void *result,
		int ld);
void starsh_ssdata_block_exp_kernel_3d(int nrows, int ncols, STARSH_int *irow,
		STARSH_int *icol, void *row_data, void *col_data, void *result,
		int ld);
void starsh_ssdata_block_exp_kernel_4d(int nrows, int ncols, STARSH_int *irow,
		STARSH_int *icol, void *row_data, void *col_data, void *result,
		int ld);
void starsh_ssdata_block_exp_kernel_nd(int nrows, int ncols, STARSH_int *irow,
		STARSH_int *icol, void *row_data, void *col_data, void *result,
		int ld);

void starsh_ssdata_block_exp_kernel_1d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_exp_kernel_2d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_exp_kernel_3d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_exp_kernel_4d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_exp_kernel_nd_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);

void starsh_ssdata_block_sqrexp_kernel_1d(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_sqrexp_kernel_2d(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_sqrexp_kernel_3d(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_sqrexp_kernel_4d(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_sqrexp_kernel_nd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);

void starsh_ssdata_block_sqrexp_kernel_1d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_sqrexp_kernel_2d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_sqrexp_kernel_3d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_sqrexp_kernel_4d_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_sqrexp_kernel_nd_simd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);


void starsh_ssdata_block_exp_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_sqrexp_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_matern_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
void starsh_ssdata_block_matern2_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);


void starsh_ssdata_block_parsimonious_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);


void starsh_ssdata_block_parsimonious2_kernel_2d_simd_gcd(int nrows, int ncols,
		STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
		void *result, int ld);
// Add definitions for other kernels after Doxygen groups have already been
// defined
#include "starsh-spatial-gsl.h"

#endif // __STARSH_SPATIAL_H__
