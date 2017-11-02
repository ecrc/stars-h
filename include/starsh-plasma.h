/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-plasma.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-22
 * */

#ifndef __STARSH_PLASMA_H__
#define __STARSH_PLASMA_H__

int solve(int n, double *A, int lda, int nrhs, double *B, int ldb,
        double *X, int ldx);

#endif // __STARSH_PLASMA_H__
