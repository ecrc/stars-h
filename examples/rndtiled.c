#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "starsh.h"
#include "starsh-rndtiled.h"

int main(int argc, char **argv)
{
    if(argc < 9)
    {
        printf("rndtiled nblocks block_size decay noise maxrank oversample "
                "tol scheme\n");
        return 1;
    }
    int nblocks = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    double decay = atof(argv[3]);
    double noise = atof(argv[4]);
    int maxrank = atoi(argv[5]);
    int oversample = atoi(argv[6]);
    double tol = atof(argv[7]);
    char *scheme = argv[8];
    int onfly = 0;
    int n = nblocks*block_size;
    int shape[2] = {n, n};
    // Generate problem by random matrices
    STARSH_rndtiled *data;
    STARSH_kernel kernel;
    starsh_rndtiled_gen(&data, &kernel, nblocks, block_size, decay, noise);
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, 2, shape, 'S', 'd', data, data, kernel,
            "Randomly generated matrix");
    starsh_problem_info(P);
    // Create new problem out of dense matrix
    Array *A;
    starsh_problem_to_array(P, &A);
    double *matrix = A->data;
    for(int i = 0; i < n; i++)
        matrix[i*(n+1)] += n;
    starsh_problem_free(P);
    starsh_problem_from_array(&P, A, 'S');
    // Init tiled cluster for tiled low-rank approximation and print info
    STARSH_cluster *C;
    //starsh_cluster_new_tiled(&C, data, n, block_size);
    starsh_cluster_new_tiled(&C, A, n, block_size);
    starsh_cluster_info(C);
    // Init tiled division into admissible blocks and print short info
    STARSH_blrf *F;
    starsh_blrf_new_tiled(&F, P, C, C, 'N');
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
    //starsh_rndtiled_free(data);
    return 0;
}
