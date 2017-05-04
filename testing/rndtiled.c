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
    if(argc != 7)
    {
        printf("%d arguments provided, but 6 are needed\n", argc-1);
        printf("rndtiled N NB decay maxrank tol scheme\n");
        return -1;
    }
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    double decay = atof(argv[3]);
    int maxrank = atoi(argv[4]);
    double tol = atof(argv[5]);
    char *scheme = argv[6];
    int oversample = 10;
    int onfly = 0;
    char symm = 'N', dtype = 'd';
    int shape[2] = {N, N};
    int info;
    // Generate problem by random matrices
    STARSH_rndtiled *data;
    STARSH_kernel kernel;
    info = starsh_application((void **)&data, &kernel, N, dtype, "rndtiled",
            "rndtiled", "nb", block_size, "decay", decay, "add_diag",
            (double)N, NULL);
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
    // Free randomly generated data
    starsh_rndtiled_free(data);
    return 0;
}
