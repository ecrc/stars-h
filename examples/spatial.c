#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "starsh.h"
#include "starsh-spatial.h"

int main(int argc, char **argv)
// Example of how to use STARS library for spatial statistics.
// For more information on STARS structures look inside of header files.
{
    if(argc < 8)
    {
        printf("%d\n", argc);
        printf("spatial.out n block_size maxrank oversample tol beta "
                "scheme\n");
        exit(0);
    }
    int n = atoi(argv[1]), block_size = atoi(argv[2]);
    int maxrank = atoi(argv[3]), oversample = atoi(argv[4]);
    double tol = atof(argv[5]), beta = atof(argv[6]);
    char *scheme = argv[7];
    int onfly = 1;
    printf("\nn=%d, bs=%d, mr=%d, os=%d, tol=%e, beta=%f, scheme=%s\n",
            n, block_size, maxrank, oversample, tol, beta, scheme);
    // Setting random seed
    srand(100);
    // Generate data for spatial statistics problem
    STARSH_ssdata *data;
    STARSH_kernel kernel;
    starsh_gen_ssdata(&data, &kernel, n, beta);
    int ndim = 2, shape[2] = {data->count, data->count}, info;
    char symm = 'S', dtype = 'd';
    // Init problem with given data and kernel
    STARSH_problem *P;
    info = starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Spatial Statistics example");
    starsh_problem_info(P);
    // Init tiled cluster for tiled low-rank approximation
    STARSH_cluster *C;
    info = starsh_cluster_new_tiled(&C, data, data->count, block_size);
    starsh_cluster_info(C);
    // Init tiled division into admissible blocks
    STARSH_blrf *F;
    info = starsh_blrf_new_tiled(&F, P, C, C, 'S');
    starsh_blrf_info(F);
    // Approximate each admissible block
    STARSH_blrm *M;
    //info = STARS_BLRM_tiled_compress_algebraic_svd(&M, F, fixrank, tol, 1);
    double time0 = omp_get_wtime();
    starsh_blrm_approximate(&M, F, maxrank, oversample, tol, onfly, scheme);
    printf("TIME TO APPROXIMATE: %e secs\n", omp_get_wtime()-time0);
    starsh_blrf_info(F);
    // 0 for onfly=0
    // Print info about approximation
    starsh_blrm_info(M);
    // Measure approximation error in Frobenius norm
    time0 = omp_get_wtime();
    printf("error, measured by starsh_blrm__dfe %e\n", starsh_blrm__dfe(M));
    printf("TIME TO MEASURE ERROR: %e secs\n", omp_get_wtime()-time0);
    //Array *A;
    /*
    info = starsh_problem_to_array(P, &A);
    Array *B;
    array_new(&B, 2, P->shape, 'd', 'F');
    starsh_blrm__dca(M, B);
    printf("info of to matrix %d\n", info);
    // Measure accuracy by dense matrices
    double diff, norm;
    info = array_diff(A, B, &diff);
    info = array_norm(A, &norm);
    printf("starsh_blrm__dca diff with Array: %e\n", diff/norm);
    */
    // Check if this problem is good for Cholesky factorization
    //printf("Info of potrf: %d\n", array_cholesky(A, 'L'));
    // Free memory, consumed by array
    //array_free(A);
    //int m = shape[0], k = 100;
    //shape[1] = k;
    //array_new(&A, 2, shape, 'd', 'F');
    //Array *resM, *resB;
    //array_new(&resM, 2, shape, 'd', 'F');
    //array_init_randn(A);
    //array_init_zeros(resM);
    //time0 = omp_get_wtime();
    //starsh_blrm__dmml(M, k, A->data, m, resM->data, m);
    //printf("TIME TO MULTIPLY BY DENSE: %e secs\n", omp_get_wtime()-time0);
    //array_dot(B, A, &resB);
    //array_diff(resM, resB, &diff);
    //array_norm(resB, &norm);
    //printf("starsh_blrm__dmml check: %e\n", diff/norm);
    //array_free(resM);
    //array_free(resB);
    //array_free(A);
    //array_free(B);
    // Free memory, used by matrix in block low-rank format
    starsh_blrm_free(M);
    // Free memory, used by block low-rank format
    starsh_blrf_free(F);
    // Free memory, used by clusterization info
    starsh_cluster_free(C);
    // Free memory, used by STARS_Problem instance
    starsh_problem_free(P);
    // */
    return 0;
}
