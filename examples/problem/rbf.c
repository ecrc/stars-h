/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file examples/problem/spatial.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#include <stdio.h>
#include <stdlib.h>
#include <starsh.h>
#include <starsh-rbf.h>

int main(int argc, char **argv)
{
    int problem_ndim = 2;
    int kernel_type = 0;
    // Correlation length
    double beta = 0.1;
    // Smoothing parameter for Matern kernel
    double nu = 0.5;
    // Set level of noise
    double noise = 0;
    // Size of desired matrix
    int N = 10370;
    // 'N' for nonsymmetric matrix and 'd' for double precision
    char symm = 'Y', dtype = 'd';
    int ndim = 3;
    STARSH_int shape[3] = {N, N, N};
    int info;
    // Generate data for spatial statistics problem
    STARSH_mddata *data;
    STARSH_kernel *kernel;
    
    starsh_generate_3d_rbf_mesh_coordinates((STARSH_mddata **)&data, "10370.txt", 10370, 3, 0, 
                          1, 0, 0.1, 0.6, 0);
    kernel=starsh_generate_3d_virus;
    STARSH_particles particles= data->particles;
    double *mesh = particles.point;  

     for (int ii =0; ii<9;ii+=3){
          printf("%f %f %f \n", mesh[ii], mesh[ii+1], mesh[ii+2]);
         }
       printf("\n");
    /*if(info != 0)
    {
        printf("wrong parameters for spatial statistics problem\n");
        return info;
    }*/
    // Init problem with given data and kernel and print short info
    STARSH_problem *problem;
    info = starsh_problem_new(&problem, ndim, shape, symm, dtype, data, data,
            kernel, "SARS-CoV-2");
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
/*    if(info != 0)
    {
        printf("Error when computing matrix elements\n");
        exit(info);
    }*/
    printf("Matrix was successfully computed\n");
    return 0;
}
