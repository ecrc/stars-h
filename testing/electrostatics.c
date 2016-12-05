#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "stars.h"
#include "stars-electrostatics.h"

int main(int argc, char **argv)
{
    if(argc < 7)
    {
        printf("%d\n", argc);
        printf("electrostatics.out row_blocks col_blocks block_size fixrank "
                "maxrank tol\n");
        exit(0);
    }
    int row_blocks = atoi(argv[1]), col_blocks = atoi(argv[2]);
    int block_size = atoi(argv[3]), fixrank = atoi(argv[4]), info;
    int maxrank = atoi(argv[5]);
    double tol = atof(argv[6]);
    printf("\nrb=%d, cb=%d, bs=%d, fr=%d, mr=%d, tol=%e\n", row_blocks,
            col_blocks, block_size, fixrank, maxrank, tol);
    // Setting random seed
    srand(time(NULL));
    // Generate data for spatial statistics problem
    STARS_esdata *data = STARS_gen_esdata(row_blocks, col_blocks, block_size);
    int ndim = 2, shape[2] = {data->count, data->count};
    char symm = 'S', dtype = 'd';
    // Init problem with given data and kernel
    STARS_Problem *P;
    info = STARS_Problem_new(&P, ndim, shape, symm, dtype, data, data,
            STARS_esdata_block_kernel, "Electrostatics example");
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
    info = STARS_BLRM_tiled_compress_algebraic_svd(&M, F, fixrank, tol, 0);
    // 0 for onfly=0
    // Print info about approximation
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(M);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Other approximation procedure
    info = STARS_BLRM_tiled_compress_algebraic_svd_ompfor(&M, F, fixrank, tol,
            0); // 0 for onfly=0
    // Print info about approximation
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(M);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Other approximation procedure
    info = STARS_BLRM_tiled_compress_algebraic_svd_batched(&M, F, fixrank, tol,
            0, maxrank, 1000000000); // 0 for onfly=0
    // Print info about approximation
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(M);
    // Get corresponding matrix
    Array *A;
    info = STARS_Problem_to_array(P, &A);
    Array *B;
    info = STARS_BLRM_to_matrix(M, &B);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Measure accuracy by dense matrices
    double diff, norm;
    info = Array_diff(A, B, &diff);
    info = Array_norm(A, &norm);
    Array_free(B);
    printf("STARS_BLRM_to_matrix diff with Array: %e\n", diff/norm);
    // 0 for onfly=0
    // Free memory, used by block low-rank format
    STARS_BLRF_free(F);
    // Free memory, used by clusterization info
    STARS_Cluster_free(C);
    // Free memory, used by spatial statistics data
    STARS_esdata_free(data);
    // Free memory, used by STARS_Problem instance
    STARS_Problem_free(P);
    // Print info about starting new problem
    printf("\nThe same, but with pregenerated matrix\n");
    Array_info(A);
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
    info = STARS_BLRM_tiled_compress_algebraic_svd(&M, F, fixrank, tol, 0);
    // Print info about approximation
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(M);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Other approximation procedure
    info = STARS_BLRM_tiled_compress_algebraic_svd_ompfor(&M, F, fixrank, tol,
            0); // 0 for onfly=0
    // Print info about approximation
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(M);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Other approximation procedure
    info = STARS_BLRM_tiled_compress_algebraic_svd_batched(&M, F, fixrank, tol,
            0, maxrank, 1000000000);
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    STARS_BLRM_error(M);
    // Get corresponding matrix
    info = STARS_BLRM_to_matrix(M, &B);
    info = Array_diff(A, B, &diff);
    info = Array_norm(A, &norm);
    Array_free(B);
    printf("STARS_BLRM_to_matrix diff with Array: %e\n", diff/norm);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Free memory, used by block low-rank format
    STARS_BLRF_free(F);
    // Free memory, used by clusterization info
    STARS_Cluster_free(C);
    // Free memory, used by STARS_Problem instance
    STARS_Problem_free(P);
    // Check if this problem is good for Cholesky factorization
    //printf("Info of potrf: %d\n", Array_Cholesky(A, 'L'));
    // Free memory, consumed by array
    Array_free(A);
    return 0;
}
