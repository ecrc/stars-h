/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/backends/sequential/kernels/dna.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-05-21
 * */

#include "common.h"
#include "starsh.h"

void starsh_kernel_dna(int nrows, int ncols, double *D, double *U, double *V,
        int *rank, int maxrank, int oversample, double tol, double *work,
        int lwork, int *iwork)
//! Fake approximation schemes, that returns rank=-1.
/*! @ingroup approximations
 * @param[in] nrows: Number of rows of a matrix.
 * @param[in] ncols: Number of columns of a matrix.
 * @param[in,out] D: Pointer to dense matrix.
 * @param[out] U: Pointer to low-rank factor `U`.
 * @param[out] V: Pointer to low-rank factor `V`.
 * @param[out] rank: Address of rank variable.
 * @param[in] maxrank: Maximum possible rank.
 * @param[in] oversample: Rank oversampling.
 * @param[in] tol: Relative error for approximation.
 * @param[in] work: Working array.
 * @param[in] lwork: Size of `work` array.
 * @param[in] iwork: Temporary integer array.
 * */
{
    *rank = -1;
}

