#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <plasma.h>

int solve(int n, double *A, int lda, double *b, double *x)
{
    cblas_dcopy(n, b, 1, x, 1);
    int info = plasma_dposv(PlasmaLower, n, 1, A, lda, x, n);
    if(info != 0)
        printf("PLASMA DPOSV INFO=%i\n", info);
    return info;
}
