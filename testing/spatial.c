#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mkl.h>
#include "starsh.h"
#include "starsh-spatial.h"
#include <stdarg.h>
#include <string.h>

int main(int argc, char **argv)
{
    if(argc != 12)
    {
        printf("%d arguments provided, but 11 are needed\n", argc-1);
        printf("spatial problem kernel beta nu N block_size scheme maxrank"
                " tol check_matvec check_cg_solve\n");
        return -1;
    }
    char *problem_type = argv[1];
    if(strcmp(problem_type, "spatial") != 0
            && strcmp(problem_type, "spatial3d") != 0)
    {
        printf("parameter problem (1st argument) must be \"spatial\" or "
                "\"spatial3d\"\n");
        return -1;
    }
    char *kernel_type = argv[2];
    double beta = atof(argv[3]);
    double nu = atof(argv[4]);
    int N = atoi(argv[5]);
    int block_size = atoi(argv[6]);
    char *scheme = argv[7];
    int maxrank = atoi(argv[8]);
    double tol = atof(argv[9]);
    int check_matvec = atoi(argv[10]);
    int check_cg_solve = atoi(argv[11]);
    int oversample = 10, onfly = 0;
    char symm = 'N', dtype = 'd';
    int ndim = 2, shape[2] = {N, N};
    int nrhs = 1;
    int info;
    srand(0);
    // Generate data for spatial statistics problem
    STARSH_ssdata *data;
    STARSH_kernel kernel;
    //starsh_gen_ssdata(&data, &kernel, n, beta);
    info = starsh_application((void **)&data, &kernel, N, dtype, problem_type,
            kernel_type, "beta", beta, "nu", nu, NULL);
    if(info != 0)
    {
        printf("Problem was NOT generated (wrong parameters)\n");
        return info;
    }
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Spatial Statistics example");
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
    starsh_blrm_approximate(&M, F, maxrank, oversample, 1e-3, onfly, scheme);
    time1 = omp_get_wtime()-time1;
    starsh_blrf_info(F);
    starsh_blrm_info(M);
    printf("TIME TO APPROXIMATE: %e secs\n", time1);
    // Measure approximation error
    time1 = omp_get_wtime();
    double rel_err = starsh_blrm__dfe(M);
    time1 = omp_get_wtime()-time1;
    printf("TIME TO MEASURE ERROR: %e secs\nRELATIVE ERROR: %e\n",
            time1, rel_err);
    // Free temporary mblr-matrix
    starsh_blrm_free(M);
    // Approximate each admissible block
    time1 = omp_get_wtime();
    starsh_blrm_approximate(&M, F, maxrank, oversample, tol, onfly, scheme);
    time1 = omp_get_wtime()-time1;
    starsh_blrf_info(F);
    starsh_blrm_info(M);
    printf("TIME TO APPROXIMATE: %e secs\n", time1);
    // Measure approximation error
    time1 = omp_get_wtime();
    rel_err = starsh_blrm__dfe(M);
    time1 = omp_get_wtime()-time1;
    printf("TIME TO MEASURE ERROR: %e secs\nRELATIVE ERROR: %e\n",
            time1, rel_err);
    // Flush STDOUT, since next step is very time consuming
    fflush(stdout);
    // Measure time for 10 BLRM matvecs and for 10 BLRM TLR matvecs
    if(check_matvec == 1)
    {
        double *x, *y, *y_tiled;
        int nrhs = 1;
        STARSH_MALLOC(x, N*nrhs);
        STARSH_MALLOC(y, N*nrhs);
        STARSH_MALLOC(y_tiled, N*nrhs);
        int iseed[4] = {0, 0, 0, 1};
        LAPACKE_dlarnv_work(3, iseed, N*nrhs, x);
        cblas_dscal(N*nrhs, 0.0, y, 1);
        cblas_dscal(N*nrhs, 0.0, y_tiled, 1);
        time1 = omp_get_wtime();
        for(int i = 0; i < 10; i++)
            starsh_blrm__dmml(M, nrhs, 1.0, x, N, 0.0, y, N);
        time1 = omp_get_wtime()-time1;
        printf("TIME FOR 10 BLRM MATVECS: %e secs\n", time1);
    }
    // Measure time for 10 BLRM and TLR matvecs and then solve with CG, initial
    // solution x=0, b is RHS and r is residual
    if(check_cg_solve == 1)
    {
        double *b, *x, *r, *CG_work;
        STARSH_MALLOC(b, N*nrhs);
        STARSH_MALLOC(x, N*nrhs);
        STARSH_MALLOC(r, N*nrhs);
        STARSH_MALLOC(CG_work, 3*(N+1)*nrhs);
        int iseed[4] = {0, 0, 0, 1};
        LAPACKE_dlarnv_work(3, iseed, N*nrhs, b);
        // Solve with CG
        time1 = omp_get_wtime();
        int info = starsh_itersolvers__dcg_omp(M, nrhs, b, N, x, N, tol,
                CG_work);
        time1 = omp_get_wtime()-time1;
        starsh_blrm__dmml(M, nrhs, -1.0, x, N, 0.0, r, N);
        cblas_daxpy(N, 1.0, b, 1, r, 1);
        double norm_rhs = cblas_dnrm2(N, b, 1);
        double norm_res = cblas_dnrm2(N, r, 1);
        printf("CG INFO=%d\nCG TIME=%f secs\nCG RELATIVE ERROR IN "
                "RHS=%e\n", info, time1, norm_res/norm_rhs);
    }
    return 0;
}
