#include <stdio.h>
#include <stdlib.h>
#include "stars.h"

int main(int argc, char **argv)
{
    int ndim = 2, nrows = 3, ncols = 3;
    int shape[ndim];
    int i, idtype, iorder;
    shape[0] = nrows;
    shape[ndim-1] = ncols;
    for(i = 1; i < ndim-1; i++)
        shape[i] = 3;
    int *irow = (int *)malloc(nrows*sizeof(int));
    for(i = 0; i < nrows; i++)
        irow[i] = i;
    int *icol = (int *)malloc(ncols*sizeof(int));
    for(i = 0; i < ncols; i++)
        icol[i] = i;
    char dtype[] = "sdcz", order[] = "FC";
    for(idtype = 0; idtype < 4; idtype++)
        for(iorder = 0; iorder < 2; iorder++)
        {
            printf("idtype %i, iorder %i\n", idtype, iorder);
            printf("dtype '%c', order '%c'\n", dtype[idtype], order[iorder]);
            Array *array = Array_new(ndim, shape, dtype[idtype],
                    order[iorder]);
            Array_init_rand(array);
            STARS_Problem *problem = STARS_Problem_from_array(array, 'N');
            Array *array2 = STARS_Problem_get_block(problem, nrows, ncols,
                    irow, icol);
            //STARS_Problem_free(problem);
            Array_info(array);
            Array_info(array2);
            //Array_print(array);
            //Array_print(array2);
            double diff = Array_diff(array, array2);
            double norm = Array_norm(array);
            Array_free(array);
            Array_free(array2);
            printf("Relative error in Frobenius norm is %f\n", diff/norm);
        }
    free(irow);
    free(icol);
}
