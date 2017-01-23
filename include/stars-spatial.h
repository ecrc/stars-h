#include <stdint.h>
#include "stars.h"

typedef struct starsh_ssdata
{
    size_t count;
    double *point;
    double beta;
} STARSH_ssdata;

int starsh_ssdata_block_exp_kernel(int nrows, int ncols, int *irow,
        int *icol, void *row_data, void *col_data, void *result);
STARSH_ssdata *starsh_gen_ssdata2(int n, double beta);
STARSH_ssdata *starsh_gen_ssdata(int row_blocks, int col_blocks,
        int block_size, double beta);
void starsh_ssdata_free(STARSH_ssdata *data);
