/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * @file minimal.c
 * @version 1.0.0.2
 * @author Aleksandr Mikhalev
 * @date 16 May 2017
 * */

#include <stdio.h>
#include <stdlib.h>
#include "starsh.h"
#include "starsh-minimal.h"

int main(int argc, char **argv)
{
    int problem_ndim = 2;
    // Since there is only one kernel for minimal, kernel_type is ignored
    int kernel_type = 0;
    // Size of desired matrix
    int N = 2500;
    // 'N' for nonsymmetric matrix and 'd' for double precision
    char symm = 'N', dtype = 'd';
    int ndim = 2, shape[2] = {N, N};
    int info;
    srand(0);
    // Generate data for spatial statistics problem
    STARSH_mindata *data;
    STARSH_kernel kernel;
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
