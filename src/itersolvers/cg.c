#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl.h>
#include "starsh.h"

int starsh_itersolvers__dcg(STARSH_blrm *M, int n, double *b, double tol,
        double *x, double *work)
{
    double *r = work;
    double *p = r+n;
    double *next_p = p+n;
    double rscheck, rsold, rsnew;
    //memset(work, 0, 3*n*sizeof(double));
    cblas_dcopy(n, b, 1, x, 1);
    starsh_blrm__dmml(M, 1, 1.0, b, n, 0.0, r, n);
    cblas_dscal(n, -1., r, 1);
    cblas_daxpy(n, 1., b, 1, r, 1);
    cblas_dcopy(n, r, 1, p, 1);
    rsold = cblas_dnrm2(n, r, 1);
    rscheck = rsold*tol;
    rsold *= rsold;
    printf("rsold=%e\n", rsold);
    for(int i = 0; i < n; i++)
    {
        starsh_blrm__dmml(M, 1, 1.0, p, n, 0.0, next_p, n);
        double tmp = cblas_ddot(n, p, 1, next_p, 1);
        double alpha = rsold/tmp;
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, next_p, 1, r, 1);
        rsnew = cblas_dnrm2(n, r, 1);
        if(rsnew < rscheck)
            break;
        rsnew *= rsnew;
        cblas_dscal(n, rsnew/rsold, p, 1);
        cblas_daxpy(n, 1., r, 1, p, 1);
        rsold = rsnew;
        printf("iter=%d rsnew=%e\n", i, rsnew);
    }
    return 0;
}
