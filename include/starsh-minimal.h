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

#ifndef __STARSH_MINIMAL_H__
#define __STARSH_MINIMAL_H__

/*! @defgroup app-mindata Minimal working example
 * @brief Minimal working example
 *
 * @ref STARSH_mindata contains only size of matrix and very simple kernel.
 * This example is intended to make it easy to understand how STARS-H works.
 *
 * @ingroup applications
 * */

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"

typedef struct starsh_mindata
//! Structure for minimal example.
/*! Main difference of this structure against others is that it does not
 * have any useful field except `count` (size of matrix). This limits possible
 * problem to those, where kernel depends @e only on row and column index.
 *
 * @ingroup app-mindata
 * */
{
    STARSH_int count;
    //!< Size of matrix.
    char dtype;
    //!< Type of matrix entry (ignored).
} STARSH_mindata;

enum STARSH_MINIMAL_KERNEL
//! List of built-in kernels for starsh_mindata_get_kernel().
/*! There is only one kernel right now, this structure is to support future
 * methods to generate synthetic matrices.
 *
 * @ingroup app-mindata
 * */
{
    STARSH_MINIMAL_KERNEL1 = 1,
    //!< The only kernel.
};

int starsh_mindata_new(STARSH_mindata **data, STARSH_int count, char dtype);
void starsh_mindata_free(STARSH_mindata *data);
int starsh_mindata_get_kernel(STARSH_kernel **kernel, STARSH_mindata *data,
        enum STARSH_MINIMAL_KERNEL type);

// KERNELS
void starsh_mindata_block_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);

#endif // __STARSH_MINIMAL_H__
