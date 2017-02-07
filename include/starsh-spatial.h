#include <stdint.h>
#include "starsh.h"

typedef struct starsh_ssdata
//! Structure for Spatial Statistics problems.
{
    size_t count;
    //!< Number of sptial points.
    double *point;
    //!< Coordinates of spatial points.
    double beta;
    //!< Parameter for exponential function.
} STARSH_ssdata;

//int starsh_ssdata_block_exp_kernel(int nrows, int ncols, int *irow,
//        int *icol, void *row_data, void *col_data, void *result);
int starsh_gen_ssdata(STARSH_ssdata **data, STARSH_kernel *kernel, int n,
        double beta);
void starsh_ssdata_free(STARSH_ssdata *data);
