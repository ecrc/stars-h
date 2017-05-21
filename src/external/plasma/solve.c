/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file src/external/plasma/solve.c
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-05-21
 * */

#include "common.h"
#include <plasma.h>

int solve(int n, double *A, int lda, int nrhs, double *B, int ldb,
        double *X, int ldx)
{
    int i;
    for(i = 0; i < nrhs; i++)
        cblas_dcopy(n, B+ldb*i, 1, X+ldx*i, 1);
    int info = plasma_dposv(PlasmaLower, n, nrhs, A, lda, X, ldx);
    if(info != 0)
        printf("PLASMA DPOSV INFO=%i\n", info);
    return info;
}
