#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "stars.h"
#include "stars-spatial.h"

int main(int argc, char **argv)
// Example of how to use STARS library for spatial statistics.
// For more information on STARS structures look inside of header files.
{
    if(argc < 8)
    {
        printf("%d\n", argc);
        printf("spatial.out n block_size fixrank maxrank tol beta rseed\n");
        exit(0);
    }
    int n = atoi(argv[1]), block_size = atoi(argv[2]), fixrank = atoi(argv[3]);
    int maxrank = atoi(argv[4]), rseed = atoi(argv[7]), info;
    double tol = atof(argv[5]), beta = atof(argv[6]);
    printf("\nn=%d, bs=%d, fr=%d, mr=%d, tol=%e, beta=%f rseed=%d\n",
            n*n, block_size, fixrank, maxrank, tol, beta, rseed);
    // Setting random seed
    srand(rseed);
    // Generate data for spatial statistics problem
    //STARS_ssdata *data = STARS_gen_ssdata(row_blocks, col_blocks, block_size,
    //        beta);
    STARS_ssdata *data = STARS_gen_ssdata2(n, beta);
    int ndim = 2, shape[2] = {data->count, data->count};
    char symm = 'S', dtype = 'd';
    // Init problem with given data and kernel
    STARS_Problem *P;
    info = STARS_Problem_new(&P, ndim, shape, symm, dtype, data, data,
            STARS_ssdata_block_exp_kernel, "Spatial Statistics example");
    STARS_Problem_info(P);
    // Init tiled cluster for tiled low-rank approximation
    STARS_Cluster *C;
    info = STARS_Cluster_new_tiled(&C, data, data->count, block_size);
    STARS_Cluster_info(C);
    // Init tiled division into admissible blocks
    STARS_BLRF *F;
    info = STARS_BLRF_new_tiled(&F, P, C, C, 'S');
    STARS_BLRF_info(F);
    // Approximate each admissible block
    STARS_BLRM *M;
    info = STARS_BLRM_tiled_compress_algebraic_svd_ompfor(&M, F, fixrank, tol,
            0); // 0 for onfly=0
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(M);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Free modified block low-rank format so that comparison is appropriate
    STARS_BLRF_free(F);
    // Get new BLRF
    STARS_BLRF_new_tiled(&F, P, C, C, 'S');
    // Approximate each admissible block
    info = STARS_BLRM_tiled_compress_algebraic_svd_batched(&M, F, fixrank, tol,
            0, maxrank, 1000000000);
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(M);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Free memory, used by block low-rank format
    STARS_BLRF_free(F);
    // Free memory, used by clusterization info
    STARS_Cluster_free(C);
    // Free memory, used by STARS_Problem instance
    STARS_Problem_free(P);
    // Free memory, used by spatial statistics data
    STARS_ssdata_free(data);
    return 0;
}
