#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>
#include "starsh.h"

int starsh_itersolvers__dcg(STARSH_blrm *M, int nrhs, double *B, int ldb,
        double *X, int ldx, double tol, double *work)
//! Conjugate gradient method for Tile low-rank matrix
/*! @param[in] M: Tile low-rank matrix.
 * @param[in] b: Right hand side.
 * @param[in] tol: Relative error threshold for residual.
 * @param[out] x: Answer.
 * @param work: Temporary array of size `3*n`.
 * */
{
    int n = M->format->problem->shape[0];
    double *R = work;
    double *P = R+n*nrhs;
    double *next_P = P+n*nrhs;
    double *rscheck = next_P+n*nrhs;
    double *rsold = rscheck+nrhs;
    double *rsnew = rsold+nrhs;
    int i;
    int finished = 0;
    starsh_blrm__dmml_omp(M, nrhs, -1.0, X, ldx, 0.0, R, n);
    for(i = 0; i < nrhs; i++)
        cblas_daxpy(n, 1., B+ldb*i, 1, R+n*i, 1);
    cblas_dcopy(n*nrhs, R, 1, P, 1);
    for(i = 0; i < nrhs; i++)
    {
        rsold[i] = cblas_dnrm2(n, R+n*i, 1);
        rscheck[i] = rsold[i]*tol;
        rsnew[i] = rscheck[i];
        rsold[i] *= rsold[i];
    }
    //printf("rsold=%e\n", rsold);
    for(i = 0; i < n; i++)
    {
        starsh_blrm__dmml_omp(M, nrhs, 1.0, P, n, 0.0, next_P, n);
        for(int j = 0; j < nrhs; j++)
        {
            if(rscheck[j] < 0)
                continue;
            double *p = P+n*j;
            double *next_p = next_P+n*j;
            double *r = R+n*j;
            double *x = X+ldx*j;
            double tmp = cblas_ddot(n, p, 1, next_p, 1);
            double alpha = rsold[j]/tmp;
            cblas_daxpy(n, alpha, p, 1, x, 1);
            cblas_daxpy(n, -alpha, next_p, 1, r, 1);
            rsnew[j] = cblas_dnrm2(n, r, 1);
            //printf("iter=%d rsnew=%e\n", i, rsnew);
            if(rsnew[j] < rscheck[j])
            {
                finished++;
                rscheck[j] = -1.;
                continue;
            }
            rsnew[j] *= rsnew[j];
            cblas_dscal(n, rsnew[j]/rsold[j], p, 1);
            cblas_daxpy(n, 1., r, 1, p, 1);
            rsold[j] = rsnew[j];
        }
        if(finished == nrhs)
            return i;
    }
    return -1;
}
