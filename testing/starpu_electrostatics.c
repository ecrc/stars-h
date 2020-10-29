/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file testing/starpu_electrostatics.c
 * @version 0.1.0
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
#include <string.h>
#include <starpu.h>
#include <starsh.h>
#include <starsh-electrostatics.h>

int main(int argc, char **argv)
{
    if(argc < 8)
    {
        printf("%d arguments provided, but 7 are needed\n", argc-1);
        printf("starpu_electrostatics ndim placement kernel N block_size "
                "maxrank tol\n");
        return 1;
    }
    int problem_ndim = atoi(argv[1]);
    int place = atoi(argv[2]);
    // Possible values can be found in documentation for enum
    // STARSH_PARTICLES_PLACEMENT
    int kernel_type = atoi(argv[3]);
    int N = atoi(argv[4]), block_size = atoi(argv[5]);
    int maxrank = atoi(argv[6]);
    double tol = atof(argv[7]);
    int onfly = 0;
    char dtype = 'd', symm = 'N';
    int ndim = 2;
    STARSH_int shape[2] = {N, N};
    int info;
    srand(0);
    // Init STARS-H
    info = starsh_init();
    if(info != 0)
        return info;
    // Generate data for electrostatics problem
    STARSH_esdata *data;
    STARSH_kernel *kernel;
    info = starsh_application((void **)&data, &kernel, N, dtype,
            STARSH_ELECTROSTATICS, kernel_type,
            STARSH_ELECTROSTATICS_NDIM, problem_ndim,
            STARSH_ELECTROSTATICS_PLACE, place, 0);
    if(info != 0)
    {
        printf("Problem was NOT generated (wrong parameters)\n");
        return info;
    }
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    info = starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Electrostatics example");
    if(info != 0)
        return info;
    starsh_problem_info(P);
    // Init tiled cluster for tiled low-rank approximation and print info
    STARSH_cluster *C;
    info = starsh_cluster_new_plain(&C, data, N, block_size);
    if(info != 0)
        return info;
    starsh_cluster_info(C);
    // Init tiled division into admissible blocks and print short info
    STARSH_blrf *F;
    STARSH_blrm *M;
    info = starsh_blrf_new_tlr(&F, P, symm, C, C);
    if(info != 0)
        return info;
    starsh_blrf_info(F);
    // Init StarPU
    (void)starpu_init(NULL);
    // Approximate each admissible block
    double time1 = omp_get_wtime();
    info = starsh_blrm_approximate(&M, F, maxrank, tol, onfly);
    if(info != 0)
        return info;
    time1 = omp_get_wtime()-time1;
    // Print info about updated format and approximation
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
        return 1;
    }
    // Measure time for 10 matvecs
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
    */
    // Deinit StarPU
    starpu_shutdown();
    return 0;
}
