#include <stdio.h>
#include <stdlib.h>
#include "stars.h"
#include "stars-spatial.h"

int main(int argc, char **argv)
{
    int n = 16;
    double beta = 0.1, tol = 1e-3;
    STARS_Problem *problem;
    STARS_BLR *blr;
    STARS_BLRmatrix *mat;
    //problem = STARS_gen_ssproblem(n, beta);
    blr = STARS_gen_ss_blrformat(n, beta);
    mat = STARS_blr__compress_algebraic_svd(blr, tol);
    return 0;
}
