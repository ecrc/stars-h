#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <mkl.h>
#include "starsh.h"
#include "starsh-spatial.h"


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if(argc < 7)
    {
        if(mpi_rank == 0)
        {
            printf("%d arguments provided, but 6 are needed\n", argc-1);
            printf("mpi sqrtn block_size kernel tol beta nu\n");
        }
        MPI_Finalize();
        exit(0);
    }
    int sqrtn = atoi(argv[1]), block_size = atoi(argv[2]);
    char *kernel_type = argv[3];
    double tol = atof(argv[4]);
    double beta = atof(argv[5]);
    double nu = atof(argv[6]);
    int maxrank = 100, oversample = 10, onfly = 0;
    char *scheme = "mpi_rsdd";
    int N = sqrtn*sqrtn;
    char symm = 'N', dtype = 'd';
    int ndim = 2, shape[2] = {N, N};
    if(mpi_rank == 0)
        printf("PARAMS: N=%d NB=%d TOL=%e\n", N, block_size, tol);
    srand(0);
    // Generate data for spatial statistics problem
    STARSH_ssdata *data;
    STARSH_kernel kernel;
    //starsh_gen_ssdata(&data, &kernel, n, beta);
    starsh_application((void **)&data, &kernel, N, dtype, "spatial",
            kernel_type, "beta", beta, "nu", nu, NULL);
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Spatial Statistics example");
    if(mpi_rank == 0)
        starsh_problem_info(P); 
    // Init tiled cluster for tiled low-rank approximation and print info
    STARSH_cluster *C;
    starsh_cluster_new_tiled(&C, data, N, block_size);
    if(mpi_rank == 0)
        starsh_cluster_info(C);
    // Init tiled division into admissible blocks and print short info
    STARSH_blrf *F;
    STARSH_blrm *M;
    //starsh_blrf_new_tiled_mpi(&F, P, C, C, symm);
    starsh_blrf_new_tiled_mpi(&F, P, C, C, symm);
    if(mpi_rank == 0)
        starsh_blrf_info(F);
    // Approximate each admissible block
    MPI_Barrier(MPI_COMM_WORLD);
    double time1 = MPI_Wtime();
    starsh_blrm_approximate(&M, F, maxrank, oversample, tol, onfly, scheme);
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime()-time1;
    if(mpi_rank == 0)
    {
        starsh_blrf_info(F);
        starsh_blrm_info(M);
    }
    if(mpi_rank == 0)
        printf("TIME TO APPROXIMATE: %e secs\n", time1);
    // Measure approximation error
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime();
    double rel_err = starsh_blrm__dfe_mpi(M);
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime()-time1;
    if(mpi_rank == 0)
        printf("TIME TO MEASURE ERROR: %e secs\nRELATIVE ERROR: %e\n",
                time1, rel_err);
    // Flush STDOUT, since next step is very time consuming
    fflush(stdout);
    // Measure time for 10 BLRM and TLR matvecs and then solve with CG, initial
    // solution x=0, b is RHS and r is residual
    double *b, *x, *r, *CG_work;
    int nrhs = 1;
    STARSH_MALLOC(b, N*nrhs);
    STARSH_MALLOC(x, N*nrhs);
    STARSH_MALLOC(r, N*nrhs);
    STARSH_MALLOC(CG_work, 3*(N+1)*nrhs);
    if(mpi_rank == 0)
    {
        int iseed[4] = {0, 0, 0, 1};
        LAPACKE_dlarnv_work(3, iseed, N*nrhs, b);
    }
    // Measure 10 TLR matvecs
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime();
    for(int i = 0; i < 10; i++)
    {
        starsh_blrm__dmml_mpi_tiled(M, nrhs, 1.0, b, N, 0.0, x, N);
    }
    time1 = MPI_Wtime()-time1;
    if(mpi_rank == 0)
    {
        printf("TIME FOR 10 TLR MATVECS: %e secs\n", time1);
    }
    // Measure 10 BLRM matvecs
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime();
    for(int i = 0; i < 10; i++)
    {
        starsh_blrm__dmml_mpi(M, nrhs, 1.0, b, N, 0.0, r, N);
    }
    time1 = MPI_Wtime()-time1;
    if(mpi_rank == 0)
    {
        printf("TIME FOR 10 BLRM MATVECS: %e secs\n", time1);
        cblas_daxpy(N, -1.0, x, 1, r, 1);
        printf("MATVEC DIFF: %e\n", cblas_dnrm2(N, r, 1)/cblas_dnrm2(N, x, 1));
        cblas_dscal(N*nrhs, 0.0, x, 1);
    }
    // Solve with CG
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime();
    int info = starsh_itersolvers__dcg_mpi(M, nrhs, b, N, x, N, tol, CG_work);
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime()-time1;
    starsh_blrm__dmml_mpi_tiled(M, nrhs, -1.0, x, N, 0.0, r, N);
    if(mpi_rank == 0)
    {
        cblas_daxpy(N, 1.0, b, 1, r, 1);
        double norm_rhs = cblas_dnrm2(N, b, 1);
        double norm_res = cblas_dnrm2(N, r, 1);
        printf("CG INFO=%d\nCG TIME=%f secs\nCG RELATIVE ERROR IN RHS=%e\n",
                info, time1, norm_res/norm_rhs);
    }
    MPI_Finalize();
    return 0;
}
