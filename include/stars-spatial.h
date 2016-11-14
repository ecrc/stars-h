#include "stars.h"

typedef struct
{
    int count;
    double *point;
    double beta;
} STARS_ssdata;

STARS_ssdata *STARS_gen_ssdata(int row_blocks, int col_blocks, int block_size,
        double beta);
void STARS_ssdata_free(STARS_ssdata *data);
STARS_Problem *STARS_gen_ssproblem(int row_blocks, int col_blocks,
        int block_size, double beta);
//STARS_BLR *STARS_gen_ss_blrformat(int row_blocks, int col_blocks,
//        int block_size, double beta);
