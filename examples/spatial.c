#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <mkl.h>
#include "starsh.h"
#include "starsh-spatial.h"

int main(int argc, char **argv)
// Example of how to use STARS library for spatial statistics.
// For more information on STARS structures look inside of header files.
{
    if(argc < 8)
    {
        printf("%d\n", argc);
        printf("spatial n block_size maxrank oversample tol beta scheme\n");
        exit(1);
    }
    int n = atoi(argv[1]), block_size = atoi(argv[2]);
    int maxrank = atoi(argv[3]), oversample = atoi(argv[4]);
    double tol = atof(argv[5]), beta = atof(argv[6]);
    char *scheme = argv[7];
    int onfly = 0;
    printf("\nn=%d, bs=%d, mr=%d, os=%d, tol=%e, beta=%f, scheme=%s\n",
            n, block_size, maxrank, oversample, tol, beta, scheme);
    // Setting random seed
    srand(100);
    // Generate data for spatial statistics problem
    STARSH_ssdata *data;
    STARSH_kernel kernel;
    starsh_gen_ssdata(&data, &kernel, n, beta);
    int ndim = 2, shape[2] = {data->count, data->count};
    char symm = 'S', dtype = 'd';
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Spatial Statistics example");
    starsh_problem_info(P);
    // Init tiled cluster for tiled low-rank approximation and print info
    STARSH_cluster *C;
    starsh_cluster_new_tiled(&C, data, data->count, block_size);
    starsh_cluster_info(C);
    // Init tiled division into admissible blocks and print short info
    STARSH_blrf *F;
    starsh_blrf_new_tiled(&F, P, C, C, symm);
    starsh_blrf_info(F);
    // Approximate each admissible block
    STARSH_blrm *M;
    double time0 = omp_get_wtime();
    starsh_blrm_approximate(&M, F, maxrank, oversample, tol, onfly, scheme);
    printf("TIME TO APPROXIMATE: %e secs\n", omp_get_wtime()-time0);
    // Print info about updated format and approximation
    starsh_blrf_info(F);
    starsh_blrm_info(M);
    // Measure approximation error in Frobenius norm
    time0 = omp_get_wtime();
    printf("error, measured by starsh_blrm__dfe %e\n", starsh_blrm__dfe(M));
    printf("TIME TO MEASURE ERROR: %e secs\n", omp_get_wtime()-time0);
    // Check conversion
    Array *A, *B;
    double norm, diff;
    starsh_problem_to_array(P, &A);
    array_new(&B, 2, P->shape, 'd', 'F');
    starsh_blrm__dca(M, B);
    array_diff(A, B, &diff);
    array_norm(A, &norm);
    printf("starsh_blrm__dca diff with Array: %e\n", diff/norm);
    // Check matvec
    double rhs[data->count], res1[data->count];
    int iseed[4] = {0, 0, 0, 1};
    LAPACKE_dlarnv_work(2, iseed, data->count, rhs);
    starsh_blrm__dmml(M, 1, 1.0, rhs, data->count, 0.0, res1, data->count);
    Array *Arhs, *Ares2;
    int k = 1, shape2[2] = {data->count, k};
    array_from_buffer(&Arhs, 2, shape2, 'd', 'F', rhs);
    //array_from_buffer(&Ares2, 1, shape, 'd', 'F', res2);
    array_dot(B, Arhs, &Ares2);
    printf("mv norms: %e %e\n", cblas_dnrm2(data->count, res1, 1),
            cblas_dnrm2(data->count, Ares2->data, 1));
    cblas_daxpy(data->count, -1., res1, 1, Ares2->data, 1);
    printf("matvec diff=%e\n", cblas_dnrm2(data->count, Ares2->data,1)/
            cblas_dnrm2(data->count, res1, 1));
    // Check CG solution
    double *x, *b, *tmp_b;
    double *CG_work;
    STARSH_MALLOC(x, data->count);
    STARSH_MALLOC(b, data->count);
    STARSH_MALLOC(tmp_b, data->count);
    STARSH_MALLOC(CG_work, 3*data->count);
    LAPACKE_dlarnv_work(3, iseed, data->count, b);
    norm = cblas_dnrm2(data->count, b, 1);
    starsh_itersolvers__dcg(M, data->count, b, tol, x, CG_work);
    starsh_blrm__dmml(M, 1, -1.0, x, data->count, 1.0, b, data->count);
    printf("result diff=%e\n", cblas_dnrm2(n, b, 1)/norm);
    // Free block low-rank matrix
    starsh_blrm_free(M);
    // Free block low-rank format
    starsh_blrf_free(F);
    // Free clusterization info
    starsh_cluster_free(C);
    // Free STARS_Problem instance
    starsh_problem_free(P);
    // Free spatial statistics randomly generated data
    starsh_ssdata_free(data);
    return 0;
}
