#include <stdio.h>
#include <stdlib.h>
#include "stars.h"
#include "stars-spatial.h"

int main(int argc, char **argv)
    // Example of how to use STARS library for spatial statistics.
    // For more information on STARS structures look inside of header files.
{
    if(argc < 6)
    {
        printf("%d\n", argc);
        printf("electrostatics.out block_size block_count maxrank tol beta\n");
        exit(0);
    }
    int block_size = atoi(argv[1]), block_count = atoi(argv[2]);
    int maxrank = atoi(argv[3]);
    double tol = atof(argv[4]), beta = atof(argv[5]);
    printf("bs=%d, bc=%d, mr=%d, tol=%e, beta=%f\n", block_size, block_count,
            maxrank, tol, beta);
    STARS_Problem *problem;
    STARS_BLR *format;
    STARS_BLRmatrix *matrix;
    format = STARS_gen_ss_blrformat(block_size, block_count, beta);
    // Problem is generated inside STARS_gen_ss_blrformat
    matrix = STARS_blr__compress_algebraic_svd(format, 15, tol);
    STARS_BLRmatrix_error(matrix);
    //STARS_BLRmatrix_info(matrix);
    STARS_BLRmatrix_free(matrix);
    //STARS_BLR_info(format);
    STARS_BLR_free(format);
    return 0;
}
