#include <stdio.h>
#include <stdlib.h>
#include "starsh.h"
#include "starsh-rndtiled.h"

int main(int argc, char **argv)
{
    int problem_ndim = 2;
    // Since there is only one kernel for rndtiled, kernel_type is ignored
    int kernel_type = 0;
    // Size of desired matrix
    int N = 2500;
    // Size of tile for tile low-rank random matrix
    int block_size = 250;
    // Singular values of each tile are 1, decay, decay^2, ...
    double decay = 0.5;
    // Addition for diagonal elements
    double diag = N;
    // 'N' for nonsymmetric matrix and 'd' for double precision
    char symm = 'N', dtype = 'd';
    int ndim = 2, shape[2] = {N, N};
    int info;
    int maxrank = 30; // Maximum rank to be used
    int oversample = 10; // Parameter for randomized SVD (extra random vectors)
    double tol = 1e-3; // Error threshold
    int onfly = 0; // Do not store dense blocks (since they are stored in data)
    char *scheme = "omp_rsdd"; // Use OpenMP randomized SVD for
                                     // compression
    srand(0);
    // Generate data for spatial statistics problem
    STARSH_rndtiled *data;
    STARSH_kernel kernel;
    // STARSH_RNDTILED for random tile low-rank matrix
    // STARSH_RNDTILED_NB to indicate next parameter shows size of tile
    // STARSH_RNDTILED_DECAY to indicate next parameter is decay of singular
    //   values
    // STARSH_RNDTILED_DIAG to indicate next parameter is addition for diagonal
    //   elements
    // 0 at the end to indicate end of arguments
    info = starsh_application((void **)&data, &kernel, N, dtype,
            STARSH_RNDTILED, kernel_type, STARSH_RNDTILED_NB, block_size,
            STARSH_RNDTILED_DECAY, decay, STARSH_RNDTILED_DIAG, diag, 0);
    if(info != 0)
    {
        printf("wrong parameters for random tile low-rank matrix\n");
        return info;
    }
    // Init problem with given data and kernel and print short info
    STARSH_problem *problem;
    info = starsh_problem_new(&problem, ndim, shape, symm, dtype, data, data,
            kernel, "Random tile low-rank matrix");
    if(info != 0)
    {
        printf("Error in STARSH problem\n");
        exit(info);
    }
    printf("STARSH problem was succesfully generated\n");
    starsh_problem_info(problem);
    // Set up clusterization (divide rows and columns into blocks)
    STARSH_cluster *cluster;
    info = starsh_cluster_new_tiled(&cluster, data, N, block_size);
    if(info != 0)
    {
        printf("Error in creation of cluster\n");
        exit(info);
    }
    // Set up format (divide matrix into tiles)
    STARSH_blrf *format;
    info = starsh_blrf_new_tiled(&format, problem, cluster, cluster, symm);
    if(info != 0)
    {
        printf("Error in creation of format\n");
        exit(info);
    }
    // Approximate with tile low-rank matrix
    STARSH_blrm *matrix;
    info = starsh_blrm_approximate(&matrix, format, maxrank, oversample, tol,
            onfly, scheme);
    if(info != 0)
    {
        printf("Error in approximation\n");
        exit(info);
    }
    printf("Done with approximation!\n");
    starsh_blrf_info(format); // Show info about format
    starsh_blrm_info(matrix); // Show info about approximation
    // Show realtive error of approximation in Frobenius norm
    double rel_err = starsh_blrm__dfe_omp(matrix);
    printf("Relative error in Frobenius norm: %e\n", rel_err);
    return 0;
}
