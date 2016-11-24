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
    int ndim = 2, shape[2] = {data->count, data->count}, info;
    char symm = 'S', dtype = 'd';
    printf("\nfiles_path=%s bs=%d mr=%d tol=%e\n", files_path, block_size,
            maxrank, tol);
    // Init problem with given data and kernel
    STARS_Problem *P;
    info = STARS_Problem_new(&P, ndim, shape, symm, dtype, data, data,
            STARS_aodata_block_kernel, "Astronomy Observation example");
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
    info = STARS_BLRM_tiled_compress_algebraic_svd_ompfor(&M, F, maxrank, tol,
            0);
    // 0 for onfly=0
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(M);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Free memory, used by block low-rank format
    STARS_BLRF_free(F);
    // Free memory, used by clusterization info
    STARS_Cluster_free(C);
    // Print info about starting new problem
    printf("\nThe same, but with pregenerated matrix\n");
    // Get corresponding matrix
    Array *A;
    info = STARS_Problem_to_array(P, &A);
    Array_info(A);
    // Free memory, used by spatial statistics data
    STARS_aodata_free(data);
    // Free memory, used by STARS_Problem instance
    STARS_Problem_free(P);
    // Set new problem from corresponding matrix
    info = STARS_Problem_from_array(&P, A, 'S');
    STARS_Problem_info(P);
    // Init tiled cluster for a new problem
    info = STARS_Cluster_new_tiled(&C, A, A->shape[0], block_size);
    STARS_Cluster_info(C);
    // Init tiled block low-rank format
    info = STARS_BLRF_new_tiled(&F, P, C, C, 'S');
    STARS_BLRF_info(F);
    // Approximate each admissible block
    info = STARS_BLRM_tiled_compress_algebraic_svd_ompfor(&M, F, maxrank, tol,
            0);
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
    // Free memory, consumed by array
    Array_free(A);
    return 0;
}

