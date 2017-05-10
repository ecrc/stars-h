#include <stdio.h>
#include <stdlib.h>
#include "starsh.h"
#include "starsh-spatial.h"

int main(int argc, char **argv)
{
    int problem_ndim = 2;
    // Possible values for kernel_type are:
    //      STARSH_SPATIAL_EXP, STARSH_SPATIAL_EXP_SIMD
    //      STARSH_SPATIAL_SQREXP, STARSH_SPATIAL_SQREXP_SIMD
    //      STARSH_SPATIAL_MATERN, STARSH_SPATIAL_MATERN_SIMD
    //      STARSH_SPATIAL_MATERN2, STARSH_SPATIAL_MATERN2_SIMD
    int kernel_type = STARSH_SPATIAL_MATERN2_SIMD;
    // Correlation length
    double beta = 0.1;
    // Smoothing parameter for Matern kernel
    double nu = 0.5;
    // Set level of noise
    double noise = 0;
    // Size of desired matrix
    int N = 2500;
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
    // Size of tile
    int block_size = 250;
    srand(0);
    // Generate data for spatial statistics problem
    STARSH_ssdata *data;
    STARSH_kernel kernel;
    // STARSH_SPATIAL for spatial statistics problem
    // kernel_type is enum type, for possible values llok into starsh-spatial.h
    // STARSH_SATIAL_NDIM to indicate next parameter shows dimensionality of
    //   spatial statistics problem
    // STARSH_SPATIAL_BETA to indicate next parameter is correlation length
    // STARSH_SPATIAL_NU to indicate next parameter is smoothing parameter for
    //   Matern kernel
    // STARSH_SPATIAL_NOISE to indicate next parameter is a noise
    // 0 at the end to indicate end of arguments
    info = starsh_application((void **)&data, &kernel, N, dtype,
            STARSH_SPATIAL, kernel_type, STARSH_SPATIAL_NDIM, problem_ndim,
            STARSH_SPATIAL_BETA, beta, STARSH_SPATIAL_NU, nu,
            STARSH_SPATIAL_NOISE, noise, 0);
    if(info != 0)
    {
        printf("wrong parameters for spatial statistics problem\n");
        return info;
    }
    // Init problem with given data and kernel and print short info
    STARSH_problem *problem;
    info = starsh_problem_new(&problem, ndim, shape, symm, dtype, data, data,
            kernel, "Spatial Statistics example");
    if(info != 0)
    {
        printf("Error in starsh problem\n");
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
