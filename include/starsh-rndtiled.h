/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-rndtiled.h
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-05-21
 * */

#ifndef __STARSH_RNDTILED_H__
#define __STARSH_RNDTILED_H__

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"

typedef struct starsh_rndtiled
//! Structure for generating synthetic BLR matrices.
{
    int n;
    //!< Number of rows of synthetic matrix.
    char dtype;
    //!< Precision of elements of a matrix.
    int nblocks;
    //! < Number of tiles in one dimension.
    int block_size;
    //!< Size of each tile.
    double *U;
    //!< Pointer to `n`-by-`block_size` matrix-generator.
    double *S;
    //!< Array of singular values, which is common for all tiles.
    double *rndS;
    //!< Array of noise in singular values for each tile.
    double add_diag;
    //!< Value to add to each diagonal element (for positive definiteness).
} STARSH_rndtiled;

enum STARSH_RNDTILED_PARAM
{
    STARSH_RNDTILED_NB = 1,
    STARSH_RNDTILED_DECAY = 2,
    STARSH_RNDTILED_DIAG = 3
};

int starsh_rndtiled_new(STARSH_rndtiled **data, int n, char dtype,
        int block_size, double decay, double add_diag);
int starsh_rndtiled_new_va(STARSH_rndtiled **data, int n, char dtype,
        va_list args);
int starsh_rndtiled_new_el(STARSH_rndtiled **data, int n, char dtype, ...);
int starsh_rndtiled_get_kernel(STARSH_kernel *kernel, STARSH_rndtiled *data,
        int type);
int starsh_rndtiled_free(STARSH_rndtiled *data);

// KERNELS
void starsh_rndtiled_block_kernel(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result);

#endif // __STARSH_RNDTILED_H__
