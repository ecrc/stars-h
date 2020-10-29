/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file examples/problem/particles.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 */

#include <stdio.h>
#include <stdlib.h>
#include <starsh.h>
#include <starsh-particles.h>

int main(int argc, char **argv)
{
    int problem_ndim = 2;
    // Size of desired matrix
    int N = 2500;
    // 'N' for nonsymmetric matrix and 'd' for double precision
    char symm = 'N', dtype = 'd';
    int ndim = 2;
    STARSH_int shape[2] = {N, N};
    int info;
    // Generate data for spatial statistics problem
    STARSH_particles *data;
    info = starsh_particles_read_from_file(&data, "particles.txt",
            STARSH_ASCII);
    if(info != STARSH_SUCCESS)
    {
        printf("INFO=%d while reading\n", info);
        return info;
    }
    info = starsh_particles_write_to_file_pointer(data, stdout,
            STARSH_ASCII); 
    if(info != STARSH_SUCCESS)
    {
        printf("INFO=%d while writing\n", info);
        return info;
    }
    starsh_particles_free(data);
    return 0;
}
