/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-minimal.h
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-13
 * */

#ifndef __STARSH_MINIMAL_H__
#define __STARSH_MINIMAL_H__

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"

typedef struct starsh_mindata
//! Minimal example for better understanding of how to deal with STARS-H.
{
    size_t count;
    char dtype;
} STARSH_mindata;

int starsh_mindata_new(STARSH_mindata **data, int n, char dtype);
int starsh_mindata_new_va(STARSH_mindata **data, int n, char dtype,
        va_list args);
int starsh_mindata_new_el(STARSH_mindata **data, int n, char dtype, ...);
void starsh_mindata_free(STARSH_mindata *data);
int starsh_mindata_get_kernel(STARSH_kernel *kernel, STARSH_mindata *data,
        int type);

// KERNELS
void starsh_mindata_block_kernel(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result);

#endif // __STARSH_MINIMAL_H__
