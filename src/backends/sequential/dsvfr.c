#include <stdio.h>
#include <stdlib.h>
#include "starsh.h"

int starsh__dsvfr(int size, double *S, double tol)
// Double precision Singular Values Frobenius norm Rank
//
// Tries ranks `size`, `size`-1, `size`-2 and so on. May be accelerated by
// binary search, but it requires additional temporary memory to be allocated.
{
    int i;
    double err_tol = 0;
    // Compute Frobenius norm by `S`
    for(i = 0; i < size; i++)
        err_tol += S[i]*S[i];
    // If all elements of S are zeros, then rank is 0
    if(err_tol == 0)
        return 0;
    // If Frobenius norm is not zero, then set rank as maximum possible value
    i = size;
    // Set tolerance
    err_tol *= tol*tol;
    double tmp_norm = S[size-1]*S[size-1];
    // Check each possible rank
    while(i > 1 && err_tol >= tmp_norm)
    {
        i--;
        tmp_norm += S[i-1]*S[i-1];
    }
    return i;
}
