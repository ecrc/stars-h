#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "stars.h"
#include "stars-electrostatics.h"

int main(int argc, char **argv)
{
    if(argc < 6)
    {
        printf("%d\n", argc);
        printf("electrostatics.out row_blocks col_blocks block_size maxrank "
                "tol\n");
        exit(0);
    }
    int row_blocks = atoi(argv[1]), col_blocks = atoi(argv[2]);
    int block_size = atoi(argv[3]), maxrank = atoi(argv[4]);
    double tol = atof(argv[5]);
    printf("\nrb=%d, cb=%d, bs=%d, mr=%d, tol=%e\n", row_blocks, col_blocks,
            block_size, maxrank, tol);
    // Setting random seed
    srand(time(NULL));
    // Generate data for spatial statistics problem
    STARS_esdata *data = STARS_gen_esdata(row_blocks, col_blocks, block_size);
    int ndim = 2, shape[2] = {data->count, data->count};
    char symm = 'S', dtype = 'd';
    // Init problem with given data and kernel
    STARS_Problem *problem = STARS_Problem_init(ndim, shape, symm, dtype,
            data, data, STARS_esdata_block_kernel, "Electrostatics example");
    STARS_Problem_info(problem);
    // Init tiled cluster for tiled low-rank approximation
    STARS_Cluster *cluster = STARS_Cluster_init_tiled(data, data->count,
            block_size);
    STARS_Cluster_info(cluster);
    // Init tiled division into admissible blocks
    STARS_BLRF *blrf = STARS_BLRF_init_tiled(problem, cluster, cluster, 'S');
    STARS_BLRF_info(blrf);
    // Approximate each admissible block
    STARS_BLRM *blrm = STARS_blrf_tiled_compress_algebraic_svd(blrf, maxrank,
            tol, 0); // 0 for onfly=0
    STARS_BLRM_info(blrm);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(blrm);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(blrm);
    // Free memory, used by block low-rank format
    STARS_BLRF_free(blrf);
    // Free memory, used by clusterization info
    STARS_Cluster_free(cluster);
    // Print info about starting new problem
    printf("\nThe same, but with pregenerated matrix\n");
    // Get corresponding matrix
    Array *array = STARS_Problem_to_array(problem);
    Array_info(array);
    // Free memory, used by spatial statistics data
    STARS_esdata_free(data);
    // Free memory, used by STARS_Problem instance
    STARS_Problem_free(problem);
    // Set new problem from corresponding matrix
    problem = STARS_Problem_from_array(array, 'S');
    STARS_Problem_info(problem);
    // Init tiled cluster for a new problem
    cluster = STARS_Cluster_init_tiled(array, array->shape[0], block_size);
    STARS_Cluster_info(cluster);
    // Init tiled block low-rank format
    blrf = STARS_BLRF_init_tiled(problem, cluster, cluster, 'S');
    STARS_BLRF_info(blrf);
    // Approximate each admissible block
    blrm = STARS_blrf_tiled_compress_algebraic_svd(blrf, maxrank, tol, 0);
    STARS_BLRM_info(blrm);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(blrm);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(blrm);
    // Free memory, used by block low-rank format
    STARS_BLRF_free(blrf);
    // Free memory, used by clusterization info
    STARS_Cluster_free(cluster);
    // Free memory, used by STARS_Problem instance
    STARS_Problem_free(problem);
    // Free memory, consumed by array
    Array_free(array);
    return 0;
}
