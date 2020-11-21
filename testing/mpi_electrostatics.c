/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file testing/mpi_electrostatics.c
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
#include <string.h>
#include <starsh.h>
#include <starsh-electrostatics.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if(argc != 8)
    {
        if(mpi_rank == 0)
        {
            printf("%d arguments provided, but 7 are needed\n", argc-1);
            printf("mpi_electrostatics ndim placement kernel N block_size "
                    "maxrank tol\n");
        }
        MPI_Finalize();
        return 1;
    }
    int problem_ndim = atoi(argv[1]);
    int place = atoi(argv[2]);
    // Possible values can be found in documentation for enum
    // STARSH_PARTICLES_PLACEMENT
    int kernel_type = atoi(argv[3]);
    int N = atoi(argv[4]);
    int block_size = atoi(argv[5]);
    int maxrank = atoi(argv[6]);
    double tol = atof(argv[7]);
    int onfly = 0;
    char symm = 'N', dtype = 'd';
    int ndim = 2;
    STARSH_int shape[2] = {N, N};
    int nrhs = 1;
    int info;
    srand(0);
    // Init STARS-H
    info = starsh_init();
    if(info != 0)
    {
        MPI_Finalize();
        return 1;
    }
    // Generate data for electrostatics problem
    STARSH_esdata *data;
    STARSH_kernel *kernel;
    //starsh_gen_ssdata(&data, &kernel, n, beta);
    info = starsh_application((void **)&data, &kernel, N, dtype,
            STARSH_ELECTROSTATICS, kernel_type,
            STARSH_ELECTROSTATICS_NDIM, problem_ndim,
            STARSH_ELECTROSTATICS_PLACE, place, 0);
    if(info != 0)
    {
        if(mpi_rank == 0)
            printf("Problem was NOT generated (wrong parameters)\n");
        MPI_Finalize();
        return 1;
    }
    // Init problem with given data and kernel and print short info
    STARSH_problem *P;
    info = starsh_problem_new(&P, ndim, shape, symm, dtype, data, data,
            kernel, "Electrostatics example");
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
        if(mpi_rank == 0)
            printf("Approximation was NOT computed due to error\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime()-time1;
    if(mpi_rank == 0)
    {
        starsh_blrf_info(F);
        starsh_blrm_info(M);
        printf("TIME TO APPROXIMATE: %e secs\n", time1);
    }
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
