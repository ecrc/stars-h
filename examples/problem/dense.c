/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file examples/problem/dense.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include <stdio.h>
#include <stdlib.h>
#include <starsh.h>

int main(int argc, char **argv)
{
    // Matrix size
    int N = 2500;
    int ndim = 2, shape[2] = {N, N}; // shape of corresponding array
    char dtype = 'd'; // 'd' for double precision array
    char order = 'F'; // Fortran order (column-major)
    char symm = 'N'; // 'S' for symmetric problem, 'N' for nonsymmetric
    int info;
    // Allocate memory for dense matrix
    double *data = malloc(N*N*sizeof(*data));
    if(data == NULL)
    {
        printf("Error in allocation of memory for data buffer\n");
        exit(1);
    }
    // Iterate over columns
    for(int j = 0; j < N; j++)
        // Iterate over rows
        for(int i = 0; i < N; i++)
        {
            // Fill matrix in Fortran order (column-major)
            data[j*N+i] = 1.0;
            // Make this matrix diagonally dominant
            if(i == j)
                data[j*N+i] += N;
        }
    // Create structure array to cover double *data
    Array *array;
    info = array_from_buffer(&array, ndim, shape, dtype, order, data);
    if(info != 0)
    {
        printf("Error in creation of Array structure\n");
        exit(info);
    }
    printf("Dense array was created succesfully\n");
    // Set up problem
    STARSH_problem *problem;
    info = starsh_problem_from_array(&problem, array, symm);
    if(info != 0)
    {
        printf("Error in generating STARS-H problem\n");
        exit(info);
    }
    printf("STARSH problem was succesfully generated\n");
    starsh_problem_info(problem);
    return 0;
}
