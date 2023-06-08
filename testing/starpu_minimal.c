/*! @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file testing/starpu_minimal.c
 * @version 0.3.1
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
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
#include <starpu.h>
#include <starsh.h>
#include <starsh-minimal.h>

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        printf("%d arguments provided, but 4 are needed\n", argc-1);
        printf("starpu_minimal N block_size maxrank tol\n");
        return 1;
    }
    int N = atoi(argv[1]), block_size = atoi(argv[2]);
    int maxrank = atoi(argv[3]);
    double tol = atof(argv[4]);
    int onfly = 0;
    char dtype = 'd', symm = 'N';
    int ndim = 2;
    STARSH_int shape[2] = {N, N};
    printf("PARAMS: N=%d NB=%d TOL=%e\n", N, block_size, tol);
    // Init STARS-H
    starsh_init();
    // Generate data for spatial statistics problem
    STARSH_mindata *data;
    STARSH_kernel *kernel;
    //starsh_gen_ssdata(&data, &kernel, n, beta);
    starsh_application((void **)&data, &kernel, N, dtype, STARSH_MINIMAL,
            STARSH_MINIMAL_KERNEL1, 0);
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Minimal example");
    starsh_problem_info(P);
    // Init tiled cluster for tiled low-rank approximation and print info
    STARSH_cluster *C;
    starsh_cluster_new_plain(&C, data, N, block_size);
    starsh_cluster_info(C);
    // Init tiled division into admissible blocks and print short info
    STARSH_blrf *F;
    STARSH_blrm *M;
    //starsh_blrf_new_tiled_mpi(&F, P, C, C, symm);
    starsh_blrf_new_tlr(&F, P, symm, C, C);
    starsh_blrf_info(F);
    // Init StarPU
    (void)starpu_init(NULL);
    // Approximate each admissible block
    double time1 = omp_get_wtime();
    starsh_blrm_approximate(&M, F, maxrank, tol, onfly);
    time1 = omp_get_wtime()-time1;
    // Deinit StarPU
    starsh_blrf_info(F);
    starsh_blrm_info(M);
    printf("TIME TO APPROXIMATE: %e secs\n", time1);
    // Measure approximation error
    time1 = omp_get_wtime();
    double rel_err = starsh_blrm__dfe(M);
    time1 = omp_get_wtime()-time1;
    printf("TIME TO MEASURE ERROR: %e secs\nRELATIVE ERROR: %e\n",
            time1, rel_err);
    if(rel_err/tol > 10.)
    {
        printf("Resulting relative error is too big\n");
        exit(1);
    }
    // Measure time for 10 BLRM matvecs and for 10 BLRM TLR matvecs
    /* Not performed due to no matvec yet with STARPU
    double *x, *y;
    int nrhs = 1;
    x = malloc(N*nrhs*sizeof(*x));
    y = malloc(N*nrhs*sizeof(*y));
    int iseed[4] = {0, 0, 0, 1};
    LAPACKE_dlarnv_work(3, iseed, N*nrhs, x);
    cblas_dscal(N*nrhs, 0.0, y, 1);
    time1 = omp_get_wtime();
    for(int i = 0; i < 10; i++)
        starsh_blrm__dmml_starpu(M, nrhs, 1.0, x, N, 0.0, y, N);
    time1 = omp_get_wtime()-time1;
    printf("TIME FOR 10 BLRM MATVECS: %e secs\n", time1);
    double norm = cblas_dnrm2(N*nrhs, y, 1);
    starsh_blrm__dmml(M, nrhs, 1.0, x, N, -1.0, y, N);
    double diff = cblas_dnrm2(N*nrhs, y, 1);
    printf("MATVEC DIFF (STARPU vs SEQUENTIAL): %f\n", diff/norm);
    */
    starpu_shutdown();
    return 0;
}
