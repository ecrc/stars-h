#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <mkl.h>
#include "starsh.h"
#include "starsh-minimal.h"


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if(argc < 4)
    {
        if(mpi_rank == 0)
        {
            printf("%d arguments provided, but 3 are needed\n", argc-1);
            printf("mpi n block_size tol\n");
        }
        MPI_Finalize();
        exit(0);
    }
    int N = atoi(argv[1]), block_size = atoi(argv[2]);
    double tol = atof(argv[3]);
    int maxrank = 100;
    int oversample = 10, onfly = 0;
    char *scheme = "mpi_rsdd";
    char dtype = 'd', symm = 'N';
    int ndim = 2, shape[2] = {N, N};
    if(mpi_rank == 0)
        printf("PARAMS: N=%d NB=%d TOL=%e\n", N, block_size, tol);
    // Generate data for spatial statistics problem
    STARSH_mindata *data;
    STARSH_kernel kernel;
    //starsh_gen_ssdata(&data, &kernel, n, beta);
    starsh_application((void **)&data, &kernel, N, dtype, "minimal", NULL,
            NULL);
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Minimal example");
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
    // Measure time for 10 BLRM matvecs and for 10 BLRM TLR matvecs
    double *x, *y, *y_tiled;
    int nrhs = 1;
    STARSH_MALLOC(x, N*nrhs);
    STARSH_MALLOC(y, N*nrhs);
    STARSH_MALLOC(y_tiled, N*nrhs);
    if(mpi_rank == 0)
    {
        int iseed[4] = {0, 0, 0, 1};
        LAPACKE_dlarnv_work(3, iseed, N*nrhs, x);
        cblas_dscal(N*nrhs, 0.0, y, 1);
        cblas_dscal(N*nrhs, 0.0, y_tiled, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime();
    for(int i = 0; i < 10; i++)
        starsh_blrm__dmml_mpi(M, nrhs, 1.0, x, N, 0.0, y, N);
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime()-time1;
    if(mpi_rank == 0)
    {
        printf("TIME FOR 10 BLRM MATVECS: %e secs\n", time1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime();
    for(int i = 0; i < 10; i++)
        starsh_blrm__dmml_mpi_tiled(M, nrhs, 1.0, x, N, 0.0, y_tiled, N);
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime()-time1;
    if(mpi_rank == 0)
    {
        cblas_daxpy(N, -1.0, y, 1, y_tiled, 1);
        printf("TIME FOR 10 TLR MATVECS: %e secs\n", time1);
        printf("MATVEC DIFF: %e\n", cblas_dnrm2(N, y_tiled, 1)
                /cblas_dnrm2(N, y, 1));
    }
    MPI_Finalize();
    return 0;
}

