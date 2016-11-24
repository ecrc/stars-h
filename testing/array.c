#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <complex.h>
#include "stars.h"

void check_SVD(int m, int n);

int main(int argc, char **argv)
{
    check_SVD(500, 500);
    /*
    Array *array, *array2;
    int shape[2] = {2, 4};
    array = Array_new(2, shape, 'd', 'F');
    Array_init_randn(array);
    array2 = Array_copy(array, 'C');
    Array_info(array);
    Array_print(array);
    Array_info(array2);
    Array_print(array2);
    */
}

void check_SVD(int m, int n)
{
    Array *A, *A2, *U, *S, *V;
    int shape[2] = {m, n};
    int idtype, iorder, info;
    char *dtype = "sdcz", *order = "FC";
    double diff, norm;
    for(idtype = 0; idtype < 4; idtype++)
    {
        for(iorder = 0; iorder < 2; iorder++)
        {
            printf("\nidtype %d, iorder %d\n", idtype, iorder);
            printf("dtype '%c', order '%c'\n", dtype[idtype], order[iorder]);
            info = Array_new(&A, 2, shape, dtype[idtype], order[iorder]);
            Array_init_randn(A);
            Array_info(A);
            info = Array_new_copy(&A2, A, 'N');
            info = Array_SVD(A2, &U, &S, &V);
            Array_free(A2);
            printf("S->dtype='%c'\n", S->dtype);
            if(S->dtype != U->dtype)
            {
                Array *S2;
                info = Array_convert(&S2, S, U->dtype);
                Array_free(S);
                S = S2;
            }
            Array_scale(U, 'C', S);
            Array_dot(U, V, &A2);
            Array_diff(A,A2, &diff);
            Array_norm(A, &norm);
            printf("SVD relative error: %e\n", diff/norm);
            Array_free(A);
            Array_free(A2);
            Array_free(U);
            Array_free(S);
            Array_free(V);
        }
    }
}
