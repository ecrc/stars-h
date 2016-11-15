#include <stdio.h>
#include "astronomy.h"
#include "stars.h"
#include "stars-astronomy.h"


int main(int argc, char **argv){

    if(argc < 5)
    {
        printf("./astronomy.out files_path block_size maxrank tol\n");
        exit(1);
    }
    char *files_path = argv[1];
    int night_idx=1;
    int snapshots_per_night=1;
    int snapshot_idx=1;
    int obs_idx=0;
    double alphaX=0.0;
    double alphaY=0.0;
    int block_size = atoi(argv[2]);
    int maxrank = atoi(argv[3]);
    double tol = atof(argv[4]);
    STARS_aodata *data = STARS_gen_aodata(files_path, night_idx,
            snapshots_per_night, snapshot_idx, obs_idx, alphaX, alphaY);
    int ndim = 2, shape[2] = {data->count, data->count};
    char symm = 'S', dtype = 'd';
    printf("\nfiles_path=%s bs=%d mr=%d tol=%e\n", files_path, block_size,
            maxrank, tol);
    // Init problem with given data and kernel
    STARS_Problem *problem = STARS_Problem_init(ndim, shape, symm, dtype,
            data, data, STARS_aodata_block_kernel, "Astronomy Observation "
            "example");
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
            tol);
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
    STARS_aodata_free(data);
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
    blrm = STARS_blrf_tiled_compress_algebraic_svd(blrf, maxrank, tol);
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

