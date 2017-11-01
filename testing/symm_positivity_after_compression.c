/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file testing/symm_positivity_after_compression.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-22
 * */

#ifdef MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "starsh.h"
#include "starsh-spatial.h"

int main(int argc, char **argv)
// Example of how to use STARS library for spatial statistics.
// For more information on STARS structures look inside of header files.
{
    /*
    if(argc < 8)
    {
        printf("%d\n", argc);
        printf("symm_positivity_after_compression n block_size maxrank "
                "oversample tol beta scheme\n");
        exit(1);
    }
    int n = atoi(argv[1]), block_size = atoi(argv[2]);
    int maxrank = atoi(argv[3]), oversample = atoi(argv[4]);
    double tol = atof(argv[5]), beta = atof(argv[6]);
    char *scheme = argv[7];
    */
    int sqrtn = 100, block_size = 1000;
    int maxrank = 100;
    double tol = 1e-9, beta = 0.1;
    double nu = 0.0;
    int onfly = 0;
    int N = sqrtn*sqrtn;
    // Setting random seed
    srand(100);
    // Init STARS-H
    starsh_init();
    // Generate data for spatial statistics problem
    STARSH_ssdata *data;
    STARSH_kernel kernel;
    int ndim = 2, shape[2] = {N, N};
    char symm = 'S', dtype = 'd';
    starsh_application((void **)&data, &kernel, N, dtype, "spatial", "exp",
            "beta", beta, "nu", nu, NULL);
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Spatial Statistics example");
    starsh_problem_info(P);
    // Init plain cluster and print info
    STARSH_cluster *C;
    starsh_cluster_new_plain(&C, data, N, block_size);
    starsh_cluster_info(C);
    // Init tlr division into admissible blocks and print short info
    STARSH_blrf *F;
    starsh_blrf_new_tlr(&F, P, C, C, symm);
    starsh_blrf_info(F);
    // Approximate each admissible block
    STARSH_blrm *M;
    double time0 = omp_get_wtime();
    starsh_blrm_approximate(&M, F, maxrank, tol, onfly);
    printf("TIME TO APPROXIMATE: %e secs\n", omp_get_wtime()-time0);
    // Print info about updated format and approximation
    starsh_blrf_info(F);
    starsh_blrm_info(M);
    // Convert to dense and run Cholesky factorization
    Array *A;
    array_new(&A, 2, P->shape, 'd', 'F');
    starsh_blrm__dca(M, A);
    printf("POTRF INFO: %d\n", array_cholesky(A, 'L'));
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
