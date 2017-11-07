/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file testing/particles.c
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <starsh.h>
#include <starsh-particles.h>

int main(int argc, char **argv)
{
    int info;
    size_t total;
    // Generate data for spatial statistics problem
    STARSH_particles *data, *data2;
    info = starsh_particles_read_from_file(&data, "particles.txt",
            STARSH_ASCII);
    if(info != STARSH_SUCCESS)
    {
        printf("starsh_particles_read_from_file(particles.txt,ASCII) "
                "INFO=%d\n", info);
        return info;
    }
    info = starsh_particles_write_to_file(data, "particles.tmp", STARSH_ASCII);
    if(info != STARSH_SUCCESS)
    {
        printf("starsh_particles_write_to_file(particles.tmp,ASCII) "
                "INFO=%d\n", info);
        return info;
    }
    info = starsh_particles_read_from_file(&data2, "particles.tmp",
            STARSH_ASCII);
    if(info != STARSH_SUCCESS)
    {
        printf("starsh_particles_read_from_file(particles.tmp,ASCII) "
                "INFO=%d\n", info);
        return info;
    }
    if(data->count != data2->count || data->ndim != data2->ndim)
    {
        printf("Error in conversion to ASCII file and back\n");
        return STARSH_FPRINTF_ERROR;
    }
    total = data->count*data->ndim*sizeof(*data->point);
    info = memcmp(data->point, data2->point, total);
    if(info != 0)
    {
        printf("Error in conversion to ASCII file and back\n");
        return STARSH_FPRINTF_ERROR;
    }
    starsh_particles_free(data2);
    info = starsh_particles_write_to_file(data, "particles.tmp",
            STARSH_BINARY);
    if(info != STARSH_SUCCESS)
    {
        printf("starsh_particles_write_to_file(particles.tmp,BINARY) "
                "INFO=%d\n", info);
        return info;
    }
    info = starsh_particles_read_from_file(&data2, "particles.tmp",
            STARSH_BINARY);
    if(info != STARSH_SUCCESS)
    {
        printf("starsh_particles_read_from_file(particles.tmp,BINARY) "
                "INFO=%d\n", info);
        return info;
    }
    if(data->count != data2->count || data->ndim != data2->ndim)
    {
        printf("Error in conversion to BINARY file and back\n");
        return STARSH_FWRITE_ERROR;
    }
    total = data->count*data->ndim*sizeof(*data->point);
    info = memcmp(data->point, data2->point, total);
    if(info != 0)
    {
        printf("Error in conversion to ASCII file and back\n");
        return STARSH_FPRINTF_ERROR;
    }
    starsh_particles_free(data2);
    starsh_particles_free(data);
    info = starsh_particles_generate(&data, 12, 2,
            STARSH_PARTICLES_RAND);
    if(info != 0)
    {
        printf("Error in generation RAND\n");
        return STARSH_UNKNOWN_ERROR;
    }
    //starsh_particles_write_to_file_pointer_ascii(data, stdout);
    starsh_particles_free(data);
    info = starsh_particles_generate(&data, 12, 2,
            STARSH_PARTICLES_RANDGRID);
    if(info != 0)
    {
        printf("Error in generation RANDGRID\n");
        return STARSH_UNKNOWN_ERROR;
    }
    //starsh_particles_write_to_file_pointer_ascii(data, stdout);
    starsh_particles_free(data);
    info = starsh_particles_generate(&data, 12, 2,
            STARSH_PARTICLES_UNIFORM);
    if(info != 0)
    {
        printf("Error in generation UNIFORM\n");
        return STARSH_UNKNOWN_ERROR;
    }
    //starsh_particles_write_to_file_pointer_ascii(data, stdout);
    starsh_particles_free(data);
    info = starsh_particles_generate(&data, 12, 2,
            STARSH_PARTICLES_QUASIUNIFORM1);
    if(info != 0)
    {
        printf("Error in generation QUASIUNIFORM1\n");
        return STARSH_UNKNOWN_ERROR;
    }
    //starsh_particles_write_to_file_pointer_ascii(data, stdout);
    starsh_particles_free(data);
    info = starsh_particles_generate(&data, 12, 2,
            STARSH_PARTICLES_QUASIUNIFORM2);
    if(info != 0)
    {
        printf("Error in generation QUASUNIFORM2\n");
        return STARSH_UNKNOWN_ERROR;
    }
    //starsh_particles_write_to_file_pointer_ascii(data, stdout);
    starsh_particles_free(data);
    return 0;
}
