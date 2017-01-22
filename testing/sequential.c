#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
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
    srand(time(NULL));
    // Generate data for spatial statistics problem
    STARS_ssdata *data = STARS_gen_ssdata2(n, beta);
    int ndim = 2, shape[2] = {data->count, data->count}, info;
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
    //info = STARS_BLRM_tiled_compress_algebraic_svd(&M, F, fixrank, tol, 1);
    double time0 = omp_get_wtime();
    if(strcmp(scheme, "sdd") == 0)
        starsh_blrm__dsdd(&M, F, tol, onfly);
    else if(strcmp(scheme, "rsdd") == 0)
        starsh_blrm__drsdd(&M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "rsdd2") == 0)
        starsh_blrm__drsdd2(&M, F, maxrank, oversample, tol, onfly);
    else if(strcmp(scheme, "qp3") == 0)
        starsh_blrm__dqp3(&M, F, maxrank, oversample, tol, onfly);
    else
    {
        printf("wrong scheme (possible: sdd, rsdd, qp3)\n");
        return 1;
    }
    printf("TIME TO APPROXIMATE: %e secs\n", omp_get_wtime()-time0);
    STARS_BLRF_info(F);
    // 0 for onfly=0
    // Print info about approximation
    STARS_BLRM_info(M);
    // Measure approximation error in Frobenius norm
    printf("error, measured by starsh_blrm__dfe %e\n", starsh_blrm__dfe(M));
    Array *A;
    info = STARS_Problem_to_array(P, &A);
    Array *B;
    Array_new(&B, 2, P->shape, 'd', 'F');
    starsh_blrm__dca(M, B);
    printf("info of to matrix %d\n", info);
    // /*
    // Measure accuracy by dense matrices
    double diff, norm;
    info = Array_diff(A, B, &diff);
    info = Array_norm(A, &norm);
    printf("starsh_blrm__dca diff with Array: %e\n", diff/norm);
    // Check if this problem is good for Cholesky factorization
    //printf("Info of potrf: %d\n", Array_Cholesky(A, 'L'));
    // Free memory, consumed by array
    Array_free(A);
    int m = shape[0], k = 100;
    shape[1] = k;
    Array_new(&A, 2, shape, 'd', 'F');
    Array *resM, *resB;
    Array_new(&resM, 2, shape, 'd', 'F');
    Array_init_randn(A);
    Array_init_zeros(resM);
    starsh_blrm__dmml(M, k, A->data, m, resM->data, m);
    Array_dot(B, A, &resB);
    Array_diff(resM, resB, &diff);
    Array_norm(resB, &norm);
    printf("starsh_blrm__dmml check: %e\n", diff/norm);
    //Array_free(resM);
    //Array_free(resB);
    //Array_free(A);
    //Array_free(B);
    // Free memory, used by matrix in block low-rank format
    STARS_BLRM_free(M);
    // Free memory, used by block low-rank format
    STARS_BLRF_free(F);
    // Free memory, used by clusterization info
    STARS_Cluster_free(C);
    // Free memory, used by STARS_Problem instance
    STARS_Problem_free(P);
    // */
    return 0;
}
