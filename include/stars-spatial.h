#include "stars.h"

typedef struct STARS_ssdata
{
    size_t count;
    double *point;
    double beta;
} STARS_ssdata;

int STARS_ssdata_block_exp_kernel(size_t nrows, size_t ncols, size_t *irow,
        size_t *icol, void *row_data, void *col_data, void *result);
STARS_ssdata *STARS_gen_ssdata(size_t row_blocks, size_t col_blocks,
        size_t block_size, double beta);
void STARS_ssdata_free(STARS_ssdata *data);
