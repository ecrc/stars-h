#include "stars.h"

typedef struct STARS_ssdata
{
    int count;
    double *point;
    double beta;
} STARS_ssdata;

int STARS_ssdata_block_exp_kernel(int nrows, int ncols, int *irow, int *icol,
        void *row_data, void *col_data, void *result);
STARS_ssdata *STARS_gen_ssdata(int row_blocks, int col_blocks, int block_size,
        double beta);
void STARS_ssdata_free(STARS_ssdata *data);
