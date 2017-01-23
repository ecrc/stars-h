#include <stdint.h>
#include "stars.h"

typedef struct starsh_ssdata
{
    size_t count;
    double *point;
    double beta;
} STARSH_ssdata;

//int starsh_ssdata_block_exp_kernel(int nrows, int ncols, int *irow,
//        int *icol, void *row_data, void *col_data, void *result);
int starsh_gen_ssdata(STARSH_ssdata **data, STARSH_kernel *kernel, int n,
        double beta);
void starsh_ssdata_free(STARSH_ssdata *data);
