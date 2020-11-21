/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file examples/problem/spatial.c
 * @version 0.3.0
 * @author Aleksandr Mikhalev
 * @date 2020-06-09
 * */

#include <stdio.h>
#include <stdlib.h>
#include <starsh.h>
#include <starsh-spatial.h>

int main(int argc, char **argv)
{
    int problem_ndim = 2;
    // Possible values for kernel_type are:
    int kernel_type = STARSH_SPATIAL_PARSIMONIOUS_SIMD;
    // Correlation length
    double beta = 0.1;
    // Smoothing parameter for Matern kernel
    double nu = 0.5;
    double nu2 = 0.5;
    // Scaling factor
    double sigma = 1.0;
    double sigma2 = 1.0;
    // Set level of noise
    double noise = 0;
    // Bivariate correlation
    double corr = 0.1;
    // Size of desired matrix
    int N = 5000;
    // 'N' for nonsymmetric matrix and 'd' for double precision
    char symm = 'N', dtype = 'd';
    int ndim = 2;
    STARSH_int shape[2] = {N, N};
    // OBSOLETE3 for kernel PARSIMONIOUS and
    // OBSOLETE4 for kernel PARSIMONIOUS2
    enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_OBSOLETE3;
    int info;
    srand(0);
    // Generate data for spatial statistics problem
    STARSH_ssdata *data;
    STARSH_kernel *kernel;
    // STARSH_SPATIAL for spatial statistics problem
    // kernel_type is enum type, for possible values look into starsh-spatial.h
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
            STARSH_SPATIAL_NU2, nu2, STARSH_SPATIAL_SIGMA, sigma,
            STARSH_SPATIAL_SIGMA2, sigma2,
            STARSH_SPATIAL_NOISE, noise, STARSH_SPATIAL_PLACE, place, 0);
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
    // Compute dense matrix
    Array *array;
    info = starsh_problem_to_array(problem, &array);
    if(info != 0)
    {
        printf("Error when computing matrix elements\n");
        exit(info);
    }
    printf("Matrix was successfully computed\n");
    return 0;
}
