#include <stdio.h>
#include <stdlib.h>
#include "stars.h"

int main(int argc, char **argv)
{
    int ndim = 2, nrows = 300, ncols = 300;
    int shape[ndim];
    int i, idtype, iorder, info;
    shape[0] = nrows;
    shape[ndim-1] = ncols;
    for(i = 1; i < ndim-1; i++)
        shape[i] = 3;
    char dtype[] = "sdcz", order[] = "FC";
    for(idtype = 0; idtype < 4; idtype++)
        for(iorder = 0; iorder < 2; iorder++)
        {
            printf("\nidtype %d, iorder %d\n", idtype, iorder);
            printf("dtype '%c', order '%c'\n", dtype[idtype], order[iorder]);
            Array *A;
            info = Array_new(&A, ndim, shape, dtype[idtype], order[iorder]);
            Array_init_rand(A);
            STARS_Problem *P;
            info = STARS_Problem_from_array(&P, A, 'N');
            Array *A2;
            info = STARS_Problem_to_array(P, &A2);
            Array_info(A);
            Array_info(A2);
            double diff, norm;
            Array_diff(A, A2, &diff);
            Array_norm(A, &norm);
            Array_free(A);
            Array_free(A2);
            printf("Relative error in Frobenius norm is %f\n", diff/norm);
        }
    return 0;
}
