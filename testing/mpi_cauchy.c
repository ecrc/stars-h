/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file testing/mpi_cauchy.c
 * @version 0.3.0
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
#include <mpi.h>
#include <starsh.h>
#include <starsh-cauchy.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if(argc < 5)
    {
        if(mpi_rank == 0)
        {
            printf("%d arguments provided, but 4 are needed\n", argc-1);
            printf("mpi_cauchy N block_size maxrank tol\n");
        }
        MPI_Finalize();
        return 1;
    }
    int N = atoi(argv[1]), block_size = atoi(argv[2]);
    int maxrank = atoi(argv[3]);
    double tol = atof(argv[4]);
    int onfly = 0;
    char dtype = 'd', symm = 'N';
    int ndim = 2;
    STARSH_int shape[2] = {N, N};
    int info;
    srand(0);
    // Init STARS-H
    info = starsh_init();
    if(info != 0)
    {
        MPI_Finalize();
        return 1;
    }
    // Generate data for spatial statistics problem
    STARSH_cauchy *data;
    STARSH_kernel *kernel;
    info = starsh_application((void **)&data, &kernel, N, dtype, STARSH_CAUCHY,
            STARSH_CAUCHY_KERNEL1, 0);
    if(info != 0)
    {
        MPI_Finalize();
        return 1;
    }
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    info = starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Cauchy example");
    if(info != 0)
    {
        MPI_Finalize();
        return 1;
    }
    if(mpi_rank == 0)
        starsh_problem_info(P); 
    // Init plain clusterization and print info
    STARSH_cluster *C;
    info = starsh_cluster_new_plain(&C, data, N, block_size);
    if(info != 0)
    {
        MPI_Finalize();
        return 1;
    }
    if(mpi_rank == 0)
        starsh_cluster_info(C);
    // Init tlr division into admissible blocks and print short info
    STARSH_blrf *F;
    STARSH_blrm *M;
    info = starsh_blrf_new_tlr_mpi(&F, P, symm, C, C);
    if(info != 0)
    {
        MPI_Finalize();
        return 1;
    }
    if(mpi_rank == 0)
        starsh_blrf_info(F);
    // Approximate each admissible block
    MPI_Barrier(MPI_COMM_WORLD);
    double time1 = MPI_Wtime();
    info = starsh_blrm_approximate(&M, F, maxrank, tol, onfly);
    if(info != 0)
    {
        MPI_Finalize();
        return 1;
    }
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
    {
        printf("TIME TO MEASURE ERROR: %e secs\nRELATIVE ERROR: %e\n",
                time1, rel_err);
        if(rel_err/tol > 10.)
        {
            printf("Resulting relative error is too big\n");
            MPI_Finalize();
            return 1;
        }
    }
    if(rel_err/tol > 10.)
    {
        MPI_Finalize();
        return 1;
    }
    // Measure time for 10 BLRM matvecs and for 10 BLRM TLR matvecs
    double *x, *y, *y_tlr;
    int nrhs = 1;
    x = malloc(N*nrhs*sizeof(*x));
    y = malloc(N*nrhs*sizeof(*y));
    y_tlr = malloc(N*nrhs*sizeof(*y_tlr));
    if(mpi_rank == 0)
    {
        int iseed[4] = {0, 0, 0, 1};
        LAPACKE_dlarnv_work(3, iseed, N*nrhs, x);
        cblas_dscal(N*nrhs, 0.0, y, 1);
        cblas_dscal(N*nrhs, 0.0, y_tlr, 1);
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
        starsh_blrm__dmml_mpi_tlr(M, nrhs, 1.0, x, N, 0.0, y_tlr, N);
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime()-time1;
    if(mpi_rank == 0)
    {
        cblas_daxpy(N, -1.0, y, 1, y_tlr, 1);
        printf("TIME FOR 10 TLR MATVECS: %e secs\n", time1);
        printf("MATVEC DIFF: %e\n", cblas_dnrm2(N, y_tlr, 1)
                /cblas_dnrm2(N, y, 1));
    }
    MPI_Finalize();
    return 0;
}

