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
#include "starsh-minimal.h"

int main(int argc, char **argv)
{
    if(argc < 6)
    {
        printf("%d arguments provided, but 5 are needed\n", argc-1);
        printf("minimal N block_size scheme maxrank tol\n");
        return -1;
    }
    int N = atoi(argv[1]), block_size = atoi(argv[2]);
    char *scheme = argv[3];
    int maxrank = atoi(argv[4]);
    double tol = atof(argv[5]);
    int oversample = 10, onfly = 0;
    char dtype = 'd', symm = 'N';
    int ndim = 2, shape[2] = {N, N};
    printf("PARAMS: N=%d NB=%d TOL=%e\n", N, block_size, tol);
    // Generate data for spatial statistics problem
    STARSH_mindata *data;
    STARSH_kernel kernel;
    //starsh_gen_ssdata(&data, &kernel, n, beta);
    starsh_application((void **)&data, &kernel, N, dtype, STARSH_MINIMAL, 0,
            0);
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Minimal example");
    starsh_problem_info(P);
    // Init tiled cluster for tiled low-rank approximation and print info
    STARSH_cluster *C;
    starsh_cluster_new_tiled(&C, data, N, block_size);
    starsh_cluster_info(C);
    // Init tiled division into admissible blocks and print short info
    STARSH_blrf *F;
    STARSH_blrm *M;
    //starsh_blrf_new_tiled_mpi(&F, P, C, C, symm);
    starsh_blrf_new_tiled(&F, P, C, C, symm);
    starsh_blrf_info(F);
    // Approximate each admissible block
    double time1 = omp_get_wtime();
    starsh_blrm_approximate(&M, F, maxrank, oversample, tol, onfly, scheme);
    time1 = omp_get_wtime()-time1;
    starsh_blrf_info(F);
    starsh_blrm_info(M);
    printf("TIME TO APPROXIMATE: %e secs\n", time1);
    // Measure approximation error
    time1 = omp_get_wtime();
    double rel_err = starsh_blrm__dfe_omp(M);
    time1 = omp_get_wtime()-time1;
    printf("TIME TO MEASURE ERROR: %e secs\nRELATIVE ERROR: %e\n",
            time1, rel_err);
    // Flush STDOUT, since next step is very time consuming
    fflush(stdout);
    // Measure time for 10 BLRM matvecs and for 10 BLRM TLR matvecs
    double *x, *y;
    int nrhs = 1;
    x = malloc(N*nrhs*sizeof(*x));
    y = malloc(N*nrhs*sizeof(*y));
    int iseed[4] = {0, 0, 0, 1};
    LAPACKE_dlarnv_work(3, iseed, N*nrhs, x);
    cblas_dscal(N*nrhs, 0.0, y, 1);
    time1 = omp_get_wtime();
    for(int i = 0; i < 10; i++)
        starsh_blrm__dmml(M, nrhs, 1.0, x, N, 0.0, y, N);
    time1 = omp_get_wtime()-time1;
    printf("TIME FOR 10 BLRM MATVECS: %e secs\n", time1);
    return 0;
}
