/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file examples/problem/randtlr.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include <stdio.h>
#include <stdlib.h>
#include <starsh.h>
#include <starsh-randtlr.h>

int main(int argc, char **argv)
{
    // Since there is only one kernel for STARSH_randtlr
    int kernel_type = STARSH_RANDTLR_KERNEL1;
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
    int ndim = 2;
    STARSH_int shape[2] = {N, N};
    int info;
    srand(0);
    // Generate data for spatial statistics problem
    STARSH_randtlr *data;
    STARSH_kernel *kernel;
    // STARSH_RANDTLR for random TLR matrix
    // STARSH_RANDTLR_NB to indicate next parameter shows size of tile
    // STARSH_RANDTLR_DECAY to indicate next parameter is decay of singular
    //   values
    // STARSH_RANDTLR_DIAG to indicate next parameter is addition for diagonal
    //   elements
    // 0 at the end to indicate end of arguments
    info = starsh_application((void **)&data, &kernel, N, dtype,
            STARSH_RANDTLR, kernel_type, STARSH_RANDTLR_NB, block_size,
            STARSH_RANDTLR_DECAY, decay, STARSH_RANDTLR_DIAG, diag, 0);
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
        printf("Error in generating STARS-H problem\n");
        exit(info);
    }
    printf("STARS-H problem was succesfully generated\n");
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
