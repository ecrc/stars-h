/*! @copyright (c) 2020 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file examples/problem/acoustic.c 
 * @version 0.3.0
 * @auther Rabab Alomairy
 * @author Aleksandr Mikhalev
 * @date 2020-06-09
 * */

#include <stdio.h>
#include <stdlib.h>
#include <starsh.h>
#include <starsh-acoustic.h>

int main(int argc, char **argv)
{
    int problem_ndim = 3; // problem dimension
    // Size of desired matrix
    int N = 7680;
    // 'N' for nonsymmetric matrix and 'd' for double precision
    char symm = 'N', dtype = 'z';
    int ndim = 2; //  tensors dimension 
    STARSH_int shape[2] = {N, N};
    int info;
    int trian =2560; // number of triangles
    int nipp = 3; // number of quads

    STARSH_acdata *data;
    STARSH_kernel *kernel;
    
    char* file_name = argv[1];
    char* file_name_interpl = argv[2];

    starsh_generate_3d_acoustic_coordinates((STARSH_acdata **)&data, N, problem_ndim, trian, nipp, 0, file_name, file_name_interpl);

    kernel=starsh_generate_3d_acoustic;

    //****Block below is not yet enabled for double complex 

    // Init problem with given data and kernel and print short info
    STARSH_problem *problem;
    info = starsh_problem_new(&problem, ndim, shape, symm, dtype, data, data,
            kernel, "acoustic-scattering");
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
