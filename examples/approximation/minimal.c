/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file examples/approximation/minimal.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include <stdio.h>
#include <stdlib.h>
#include <starsh.h>
#include <starsh-minimal.h>

int main(int argc, char **argv)
{
    int problem_ndim = 2;
    // Since there is only one kernel for minimal example
    int kernel_type = STARSH_MINIMAL_KERNEL1;
    // Size of desired matrix
    int N = 2500;
    // 'N' for nonsymmetric matrix and 'd' for double precision
    char symm = 'N', dtype = 'd';
    int ndim = 2;
    STARSH_int shape[2] = {N, N};
    int info;
    int maxrank = 30; // Maximum rank to be used
    int oversample = 10; // Parameter for randomized SVD (extra random vectors)
    double tol = 1e-3; // Error threshold
    int onfly = 1; // Do not store dense blocks (since they are stored in data)
    // Size of tile
    int block_size = 250;
    srand(0);
    // Init STARS-H
    starsh_init();
    // Generate data for spatial statistics problem
    STARSH_mindata *data;
    STARSH_kernel *kernel;
    // STARSH_MINIMAL for random tile low-rank matrix
    // minimal problem does not have any parameters
    // 0 at the end to indicate end of arguments
    info = starsh_application((void **)&data, &kernel, N, dtype,
            STARSH_MINIMAL, kernel_type, 0);
    if(info != 0)
    {
        printf("wrong parameters for minimal example\n");
        return info;
    }
    // Init problem with given data and kernel and print short info
    STARSH_problem *problem;
    info = starsh_problem_new(&problem, ndim, shape, symm, dtype, data, data,
            kernel, "Minimal example");
    if(info != 0)
    {
        printf("Error in STARSH problem\n");
        exit(info);
    }
    printf("STARSH problem was succesfully generated\n");
    starsh_problem_info(problem);
    // Set up clusterization (divide rows and columns into blocks)
    STARSH_cluster *cluster;
    info = starsh_cluster_new_plain(&cluster, data, N, block_size);
    if(info != 0)
    {
        printf("Error in creation of cluster\n");
        exit(info);
    }
    // Set up format (divide matrix into tiles)
    STARSH_blrf *format;
    info = starsh_blrf_new_tlr(&format, problem, symm, cluster, cluster);
    if(info != 0)
    {
        printf("Error in creation of format\n");
        exit(info);
    }
    // Approximate with tile low-rank matrix
    STARSH_blrm *matrix;
    info = starsh_blrm_approximate(&matrix, format, maxrank, tol,
            onfly);
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
