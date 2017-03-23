#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "starsh.h"
#include "starsh-spatial.h"


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if(argc < 5)
    {
        printf("%d\n", argc);
        printf("mpi sqrtn block_size kernel seed\n");
        exit(1);
    }
    int sqrtn = atoi(argv[1]), block_size = atoi(argv[2]);
    char *kernel_type = argv[3];
    int randseed = atoi(argv[4]);
    srand(randseed);
    double beta = 0.1;
    double nu = 0.5;
    int maxrank = 100, oversample = 10, onfly = 1;
    double tol = 1e-12;
    char *scheme = "mpi_rsdd";
    int N = sqrtn*sqrtn;
    char symm = 'S', dtype = 'd';
    int ndim = 2, shape[2] = {N, N};
    srand(100);
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
    starsh_blrf_new_tiled_mpi(&F, P, C, C, symm);
    if(mpi_rank == 0)
        starsh_blrf_info(F);
    // Approximate each admissible block
    MPI_Barrier(MPI_COMM_WORLD);
    double time1 = MPI_Wtime();
    starsh_blrm_approximate(&M, F, maxrank, oversample, tol, onfly, scheme);
    time1 = MPI_Wtime()-time1;
    if(mpi_rank == 0)
        starsh_blrf_info(F);
    //starsh_blrm_info(M);
    if(mpi_rank == 0)
        printf("TIME TO APPROXIMATE: %e secs\n", time1);
    double rel_err = starsh_blrm__dfe_mpi(M);
//    if(mpi_rank == 0)
        printf("RELATIVE ERROR: %e\n", rel_err);
    MPI_Finalize();
    return 0;
}
