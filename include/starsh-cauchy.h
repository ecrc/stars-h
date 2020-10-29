/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-minimal.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __STARSH_CAUCHY_H__
#define __STARSH_CAUCHY_H__

/*! @defgroup app-cauchy Cauchy example
 * @brief Cauchy example
 * @details Uses @ref STARSH_particles structure to hold locations of points
 * (1-dimensional) and values for diagonal elements of corresponding matrix.
 *
 * @ingroup applications
 * */

// Add definitions for size_t, va_list and STARSH_kernel.
#include "starsh.h"
// Add definitions for STARSH_particles.
#include "starsh-particles.h"

//! Cauchy problem reuses structure for particles.
//! @ingroup app-cauchy
typedef STARSH_particles STARSH_cauchy;

enum STARSH_CAUCHY_KERNEL
//! List of built-in kernels for starsh_cauchy_get_kernel().
/*! There is only one kernel right now, this structure is to support future
 * methods to generate synthetic matrices.
 *
 * @ingroup app-mindata
 * */
{
    STARSH_CAUCHY_KERNEL1 = 1,
    //!< The only kernel.
};

enum STARSH_CAUCHY_PARAM
//! List of parameters for starsh_application().
/*! In the table below each constant corresponds to a given argument and type
 * for starsh_cauchy_new(). These constants are used to generate problem
 * with incomplete set of parameters via starsh_application() or
 * starsh_cauchy_new_va().
 *
 * @sa starsh_application(), starsh_cauchy_new(), starsh_cauchy_init(),
 *      starsh_cauchy_new_va().
 * @ingroup app-spatial
 * */
{
    STARSH_CAUCHY_POINT = 1,
    //!< Values of `X_i` for Cauchy matrix.
    STARSH_CAUCHY_DIAG = 2,
    //!< Diagonal values of Cauchy matrix.
};

int starsh_cauchy_init(STARSH_cauchy **data, STARSH_int count, double *point);
int starsh_cauchy_new(STARSH_cauchy **data, STARSH_int count, double *point,
        double *diag);
int starsh_cauchy_new_va(STARSH_cauchy **data, STARSH_int count, va_list args);
int starsh_cauchy_new_el(STARSH_cauchy **data, STARSH_int count, ...);
void starsh_cauchy_free(STARSH_cauchy *data);
int starsh_cauchy_get_kernel(STARSH_kernel **kernel, STARSH_cauchy *data,
        enum STARSH_CAUCHY_KERNEL type);

// KERNELS
void starsh_cauchy_block_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);

#endif // __STARSH_CAUCHY_H__
