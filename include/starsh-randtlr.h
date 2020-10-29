/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-randtlr.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __STARSH_RANDTLR_H__
#define __STARSH_RANDTLR_H__

/*! @defgroup app-randtlr Random TLR matrix
 * @brief Synthetic random TLR matrix routines
 *
 * @ref STARSH_randtlr is about generation of TLR matrix in a special form.
 * Each tile is generated as a multiplication of 3 matrices: \f$ U_i \f$,
 * \f$ S \f$ and \f$ U_j \f$, where \f$ i \f$ and \f$ j \f$ are row and column
 * indexes of some orthogonal matrix \f$ U \f$ and \f$ S \f$ is a diagonal
 * matrix.
 *
 * @ingroup applications
 * */

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"

typedef struct starsh_randtlr
//! Structure for synthetic TLR matrices.
/*! @ingroup app-randtlr
 * */
{
    STARSH_int count;
    //!< Number of rows/columns of synthetic matrix.
    //char dtype; // not supported yet
    ////!< Precision of elements of a matrix.
    STARSH_int nblocks;
    //!< Number of tiles in one dimension.
    STARSH_int block_size;
    //!< Size of each tile.
    double *U;
    //!< Pointer to `count`-by-`block_size` matrix-generator.
    double *S;
    //!< Array of singular values, which is common for all tiles.
    double diag;
    //!< Value to add to each diagonal element (for positive definiteness).
} STARSH_randtlr;

enum STARSH_RANDTLR_KERNEL
//! List of built-in kernels for starsh_randtlr_get_kernel().
/*! There is only one kernel right now, this structure is to support future
 * methods to generate synthetic matrices.
 *
 * @ingroup app-randtlr
 * */
{
    STARSH_RANDTLR_KERNEL1 = 1,
    //!< The only kernel.
};

enum STARSH_RANDTLR_PARAM
//! List of parameters for starsh_application().
/*! In the table below each constant correspond to a given argument and type
 * for starsh_randtlr_generate(). These constants are used to generate problem
 * with incomplete set of parameters via starsh_application(),
 * starsh_randtlr_generate_va() or starsh_randtlr_generate_el().
 *
 * @sa starsh_application(), starsh_randtlr_generate(),
 *      starsh_randtlr_generate_va(), starsh_randtlr_generate_el().
 * @ingroup app-randtlr
 * */
{
    STARSH_RANDTLR_NB = 1,
    //!< Size of tiles.
    STARSH_RANDTLR_DECAY = 2,
    //!< Decay of singular values, first singular value is 1.0.
    STARSH_RANDTLR_DIAG = 3,
    //!< Value to add to diagonal elements.
};

int starsh_randtlr_generate(STARSH_randtlr **data, STARSH_int count,
        STARSH_int block_size, double decay, double diag);
int starsh_randtlr_generate_va(STARSH_randtlr **data, STARSH_int count,
        va_list args);
int starsh_randtlr_generate_el(STARSH_randtlr **data, STARSH_int count, ...);
int starsh_randtlr_get_kernel(STARSH_kernel **kernel, STARSH_randtlr *data,
        enum STARSH_RANDTLR_KERNEL type);
void starsh_randtlr_free(STARSH_randtlr *data);

// KERNELS
void starsh_randtlr_block_kernel(int nrows, int ncols, STARSH_int *irow,
        STARSH_int *icol, void *row_data, void *col_data, void *result,
        int ld);

#endif // __STARSH_RANDTLR_H__
