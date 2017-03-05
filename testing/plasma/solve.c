#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <omp.h>
#include "starsh.h"
#include "starsh-spatial.h"
#include "starsh-plasma.h"
#include <plasma.h>

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        printf("%d\n", argc);
        printf("spatial n block_size beta\n");
        exit(1);
    }
    int n = atoi(argv[1]), block_size = atoi(argv[2]);
    double beta = atof(argv[3]);
    int ncores = atoi(argv[4]);
    int maxrank = 100, oversample = 10, onfly = 0;
    double tol = 1e-12;
    char *scheme = "omp_rsdd";
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
    // Get array from problem
    Array *A;
    double time0 = omp_get_wtime();
    starsh_problem_to_array(P, &A);
    time0 = omp_get_wtime()-time0;
    printf("Compute entire matrix: %e secs\n", time0);
    int N = A->shape[0];
    double *b, *x, *CG_work;
    STARSH_MALLOC(b, N);
    STARSH_MALLOC(x, N);
    int iseed[4] = {0, 0, 0, 1};
    LAPACKE_dlarnv_work(3, iseed, N, b);
    //LAPACKE_dlarnv_work(3, iseed, N, x);
    STARSH_MALLOC(CG_work, 3*N);
    plasma_init(ncores);
    time0 = omp_get_wtime();
    solve(N, A->data, N, b, x);
    time0 = omp_get_wtime()-time0;
    printf("Time to solve SPD problem: %e secs\n", time0);
    plasma_finalize();
    // Init tiled cluster for tiled low-rank approximation and print info
    time0 = omp_get_wtime();
    STARSH_cluster *C;
    starsh_cluster_new_tiled(&C, data, data->count, block_size);
    //starsh_cluster_info(C);
    // Init tiled division into admissible blocks and print short info
    STARSH_blrf *F;
    starsh_blrf_new_tiled(&F, P, C, C, symm);
    //starsh_blrf_info(F);
    // Approximate each admissible block
    STARSH_blrm *M;
    double time1 = omp_get_wtime();
    starsh_blrm_approximate(&M, F, maxrank, oversample, tol, onfly, scheme);
    time1 = omp_get_wtime()-time1;
    printf("TIME TO APPROXIMATE: %e secs\n", time1);
    // Solve with CG, approximate solution is in x, initial guess is zero
    //memset(x, 0, data->count*sizeof(double));
    cblas_dcopy(N, b, 1, x, 1);
    time1 = omp_get_wtime();
    starsh_itersolvers__dcg(M, b, tol, x, CG_work);
    double time2 = omp_get_wtime();
    printf("TIME TO SOLVE: %e secs\n", time2-time1);
    printf("TOTAL TIME FOR STARSH: %e secs\n", time2-time0);
    return 0;
}
