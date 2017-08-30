/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file testing/rndtiled.c
 * @version 1.0.0
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
#include "starsh-rndtiled.h"

int main(int argc, char **argv)
{
    if(argc != 6)
    {
        printf("%d arguments provided, but 5 are needed\n", argc-1);
        printf("rndtiled N NB decay maxrank tol\n");
        return -1;
    }
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    double decay = atof(argv[3]);
    int maxrank = atoi(argv[4]);
    double tol = atof(argv[5]);
    int onfly = 0;
    char symm = 'N', dtype = 'd';
    int shape[2] = {N, N};
    int info;
    // Init STARS-H
    starsh_init();
    // Generate problem by random matrices
    STARSH_rndtiled *data;
    STARSH_kernel kernel;
    info = starsh_application((void **)&data, &kernel, N, dtype,
            STARSH_RNDTILED, 0, STARSH_RNDTILED_NB, block_size,
            STARSH_RNDTILED_DECAY, decay, STARSH_RNDTILED_DIAG,
            (double)N, 0);
    if(info != 0)
    {
        printf("Problem was NOT generated (wrong parameters)\n");
        return info;
    }
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, 2, shape, symm, dtype, data, data, kernel,
            "Randomly generated tiled blr-matrix");
    starsh_problem_info(P);
    // Init tiled cluster for tiled low-rank approximation and print info
    STARSH_cluster *C;
    starsh_cluster_new_tiled(&C, data, N, block_size);
    starsh_cluster_info(C);
    // Init tiled division into admissible blocks and print short info
    STARSH_blrf *F;
    starsh_blrf_new_tiled(&F, P, C, C, symm);
    starsh_blrf_info(F);
    // Approximate each admissible block
    STARSH_blrm *M;
    double time1 = omp_get_wtime();
    starsh_blrm_approximate(&M, F, maxrank, tol, onfly);
    time1 = omp_get_wtime()-time1;
    // Print info about updated format and approximation
    starsh_blrf_info(F);
    starsh_blrm_info(M);
    printf("TIME TO APPROXIMATE: %e secs\n", time1);
    // Measure approximation error
    time1 = omp_get_wtime();
    double rel_err = starsh_blrm__dfe_omp(M);
    time1 = omp_get_wtime()-time1;
    printf("TIME TO MEASURE ERROR: %e secs\nRELATIVE ERROR: %e\n",
            time1, rel_err);
    if(rel_err/tol > 10.)
    {
        printf("Resulting relative error is too big\n");
        exit(1);
    }
    // Free block low-rank matrix
    starsh_blrm_free(M);
    // Free block low-rank format
    starsh_blrf_free(F);
    // Free clusterization info
    starsh_cluster_free(C);
    // Free STARS_Problem instance
    starsh_problem_free(P);
    // Free randomly generated data
    starsh_rndtiled_free(data);
    return 0;
}
