#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "stars.h"
#include "stars-electrostatics.h"

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        printf("%d\n", argc);
        printf("electrostatics.out block_size block_count maxrank tol\n");
        exit(0);
    }
    int block_size = atoi(argv[1]), block_count = atoi(argv[2]);
    int maxrank = atoi(argv[3]);
    double tol = atof(argv[4]);
    printf("bs=%d, bc=%d, mr=%d, tol=%e\n", block_size, block_count, maxrank,
            tol);
    STARS_Problem *problem;
    STARS_BLR *format;
    STARS_BLRmatrix *matrix;
    format = STARS_gen_es_blrformat(block_size, block_count);
    // Problem is generated inside STARS_gen_ss_blrformat
    matrix = STARS_blr__compress_algebraic_svd(format, maxrank, tol);
    STARS_BLRmatrix_error(matrix);
    //STARS_BLRmatrix_info(matrix);
    STARS_BLRmatrix_free(matrix);
    //STARS_BLR_info(format);
    STARS_BLR_free(format);
    return 0;
}
