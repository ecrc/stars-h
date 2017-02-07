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
        printf("spatial n block_size maxrank oversample tol beta scheme\n");
        exit(1);
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
    starsh_blrf_new_tiled(&F, P, C, C, 'S');
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
