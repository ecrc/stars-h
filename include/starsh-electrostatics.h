/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-electrostatics.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __STARSH_ELECTROSTATICS_H__
#define __STARSH_ELECTROSTATICS_H__

/*! @defgroup app-electrostatics Electrostatics problem
 * @ingroup applications
 * @brief Template for electrostatics problems.
 *
 * @ref STARSH_esdata holds all the necessary data.
 * */

/*! @defgroup app-electrostatics-kernels Kernels
 * @ingroup app-electrostatics
 * @brief Set of kernels for electrostatics problems
 *
 * Click on functions to view implemented equations.
 * */

// Add definitions for size_t, va_list, STARSH_kernel and STARSH_particles
#include "starsh.h"
#include "starsh-particles.h"

//! Electrostatics problem reuses structure for particles
typedef STARSH_particles STARSH_esdata; 

enum STARSH_ELECTROSTATICS_KERNEL
//! List of built-in kernels for starsh_esdata_get_kernel().
/*! For more info on exact formulas inside kernels, take a look at functions
 * starsh_esdata_block_coulomb_potential_kernel_nd(),
 * starsh_esdata_block_coulomb_potential_kernel_nd_simd().
 *
 * @ingroup app-electrostatics
 * */
{
    STARSH_ELECTROSTATICS_COULOMB_POTENTIAL = 1,
    /*!< Coulomb potential kernel.
     * @sa starsh_esdata_block_coulomb_potential_kernel_nd().
     * */
    STARSH_ELECTROSTATICS_COULOMB_POTENTIAL_SIMD = 2,
    /*!< Coulomb potential kernel.
     * @sa starsh_esdata_block_coulomb_potential_kernel_nd_simd().
     * */
};

enum STARSH_ELECTROSTATICS_PARAM
//! List of parameters for starsh_application().
/*! In the table below each constant corresponds to a given argument and type
 * for starsh_esdata_generate(). These constants are used to generate problem
 * with incomplete set of parameters via starsh_application(),
 * starsh_esdata_generate_va() or starsh_esdata_generate_el().
 *
 * @sa starsh_application(), starsh_esdata_generate(),
 *      starsh_esdata_generate_va(), starsh_esdata_generate_el().
 * @ingroup app-electrostatics
 * */
{
    STARSH_ELECTROSTATICS_NDIM = 1,
    //!< Dimensionality of space (`ndim`, integer).
    STARSH_ELECTROSTATICS_PLACE = 2,
    //!< Distribution of particles (`place`, @ref STARSH_PARTICLES_PLACEMENT).
};

int starsh_esdata_new(STARSH_esdata **data, STARSH_int count, int ndim);
int starsh_esdata_init(STARSH_esdata **data, STARSH_int count, int ndim,
        double *point);
int starsh_esdata_generate(STARSH_esdata **data, STARSH_int count, int ndim,
        enum STARSH_PARTICLES_PLACEMENT place);
int starsh_esdata_generate_va(STARSH_esdata **data, STARSH_int count,
        va_list args);
int starsh_esdata_generate_el(STARSH_esdata **data, STARSH_int count, ...);
int starsh_esdata_get_kernel(STARSH_kernel **kernel, STARSH_esdata *data,
         enum STARSH_ELECTROSTATICS_KERNEL type);
void starsh_esdata_free(STARSH_esdata *data);

// KERNELS

void starsh_esdata_block_coulomb_potential_kernel_1d(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_esdata_block_coulomb_potential_kernel_2d(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_esdata_block_coulomb_potential_kernel_3d(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_esdata_block_coulomb_potential_kernel_4d(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_esdata_block_coulomb_potential_kernel_nd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);

void starsh_esdata_block_coulomb_potential_kernel_1d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_esdata_block_coulomb_potential_kernel_2d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_esdata_block_coulomb_potential_kernel_3d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_esdata_block_coulomb_potential_kernel_4d_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);
void starsh_esdata_block_coulomb_potential_kernel_nd_simd(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int ld);

#endif // __STARSH_ELECTROSTATICS_H__

