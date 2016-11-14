#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "stars.h"
#include "stars-spatial.h"

int main(int argc, char **argv)
// Example of how to use STARS library for spatial statistics.
// For more information on STARS structures look inside of header files.
{
    if(argc < 8)
    {
        printf("%d\n", argc);
        printf("spatial.out row_blocks col_blocks block_size maxrank "
                "tol beta heatmap-filename\n");
        exit(0);
    }
    int row_blocks = atoi(argv[1]), col_blocks = atoi(argv[2]);
    int block_size = atoi(argv[3]), maxrank = atoi(argv[4]);
    double tol = atof(argv[5]), beta = atof(argv[6]);
    char *heatmap_fname = argv[7];
    printf("rb=%d, cb=%d, bs=%d, mr=%d, tol=%e, beta=%f\n",
            row_blocks, col_blocks, block_size, maxrank, tol, beta);
    STARS_Problem *problem;
    STARS_BLRF *blrf;
    STARS_BLRM *blrm;
    problem = STARS_gen_ssproblem(row_blocks, col_blocks, block_size, beta);
    blrf = STARS_BLRF_tiled(problem, 'S', block_size);
    blrm = STARS_blrf_tiled_compress_algebraic_svd(blrf, maxrank, tol);
    STARS_BLRM_error(blrm);
    STARS_BLRM_free(blrm);
    STARS_BLRF_free(blrf);
    STARS_ssdata_free(problem->row_data);
    STARS_Problem_free(problem);
    return 0;
}
