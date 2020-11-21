/*! @copyright (c) 2020 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file examples/problem/rbf_cube.c
 * @version 0.3.0
 * @auther Rabab Alomairy
 * @author Aleksandr Mikhalev
 * @date 2020-06-09
 * */

#include <stdio.h>
#include <stdlib.h>
#include <starsh.h>
#include <starsh-rbf.h>

int main(int argc, char **argv)
{
    int problem_ndim = 3; // problem dimension
    int kernel_type = 0;
    // Size of desired matrix
    int N = 10370;
    // 'N' for nonsymmetric matrix and 'd' for double precision
    char symm = 'S', dtype = 'd';
    int ndim = 2; //  tensors dimension 
    STARSH_int shape[2] = {N, N};
    int info;
    int isreg = 1; // it is either 0 or 1 if you want to add regularizer
    double reg = 1.1; // regularization value
    int ordering = 0; // 0: no ordering, 1: Morton ordering
    double rad = 0.6; //RBF scaling factor 

    // Generate data for mesh deformation problem
    STARSH_mddata *data;
    STARSH_kernel *kernel;
    
    starsh_generate_3d_rbf_mesh_coordinates_cube((STARSH_mddata **)&data, N, problem_ndim, 
                                             kernel_type, isreg, reg, rad, ordering);

    kernel=starsh_generate_3d_cube;
    STARSH_particles particles= data->particles;
    double *mesh = particles.point;  

    // Init problem with given data and kernel and print short info
    STARSH_problem *problem;
    info = starsh_problem_new(&problem, ndim, shape, symm, dtype, data, data,
            kernel, "Cube");
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
