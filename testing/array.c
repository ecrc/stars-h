#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <complex.h>
#include "stars.h"

void check_SVD(int m, int n);

int main(int argc, char **argv)
{
    check_SVD(2, 20);
}

void check_SVD(int m, int n)
{
    Array *array, *array2, *U, *S, *V;
    int shape[2] = {m, n};
    int idtype, iorder;
    char *dtype = "sdcz", *order = "FC";
    for(idtype = 0; idtype < 4; idtype++)
    {
        for(iorder = 0; iorder < 2; iorder++)
        {
            printf("idtype %i, iorder %i\n", idtype, iorder);
            printf("dtype '%c', order '%c'\n", dtype[idtype], order[iorder]);
            array = Array_new(2, shape, dtype[idtype], order[iorder]);
            Array_init_randn(array);
            array2 = Array_copy(array);
            Array_SVD(array2, &U, &S, &V);
            Array_free(array2);
            printf("S->dtype='%c'\n", S->dtype);
            if(S->dtype != U->dtype)
            {
                Array *S2;
                S2 = Array_convert(S, U->dtype);
                Array_free(S);
                S = S2;
            }
            Array_scale(U, 'C', S);
            array2 = Array_dot(U, V);
            printf("SVD relative error: %e\n", Array_error(array, array2));
            Array_free(array);
            Array_free(array2);
            Array_free(U);
            Array_free(S);
            Array_free(V);
        }
    }
}
