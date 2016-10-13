#include "stars.h"

typedef struct
{
    int count;
    double *point;
    double beta;
} STARS_ssdata;

STARS_Problem *STARS_gen_ssproblem(int row_blocks, int col_blocks,
        int block_size, double beta);
STARS_BLR *STARS_gen_ss_blrformat(int row_blocks, int col_blocks,
        int block_size, double beta);
