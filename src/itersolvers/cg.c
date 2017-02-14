#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>
#include "starsh.h"

int starsh_itersolvers__dcg(STARSH_blrm *M, double *b, double tol,
        double *x, double *work)
//! Conjugate gradient method for Tile low-rank matrix
/*! @param[in] M: Tile low-rank matrix.
 * @param[in] b: Right hand side.
 * @param[in] tol: Relative error threshold for residual.
 * @param[out] x: Answer.
 * @param work: Temporary array of size `3*n`.
 * */
{
    int n = M->format->problem->shape[0];
    double *r = work;
    double *p = r+n;
    double *next_p = p+n;
    double rscheck, rsold, rsnew;
    starsh_blrm__dmml(M, 1, -1.0, x, n, 0.0, r, n);
    cblas_daxpy(n, 1., b, 1, r, 1);
    cblas_dcopy(n, r, 1, p, 1);
    rsold = cblas_dnrm2(n, r, 1);
    rscheck = rsold*tol;
    rsold *= rsold;
    //printf("rsold=%e\n", rsold);
    for(int i = 0; i < n; i++)
    {
        starsh_blrm__dmml(M, 1, 1.0, p, n, 0.0, next_p, n);
        double tmp = cblas_ddot(n, p, 1, next_p, 1);
        double alpha = rsold/tmp;
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, next_p, 1, r, 1);
        rsnew = cblas_dnrm2(n, r, 1);
        //printf("iter=%d rsnew=%e\n", i, rsnew);
        if(rsnew < rscheck)
            break;
        rsnew *= rsnew;
        cblas_dscal(n, rsnew/rsold, p, 1);
        cblas_daxpy(n, 1., r, 1, p, 1);
        rsold = rsnew;
    }
    return 0;
}
