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
