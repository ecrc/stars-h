/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-particles.h
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2017-08-22
 * */

#ifndef __STARSH_PARTICLES_H__
#define __STARSH_PARTICLES_H__

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"
// Add definitions for FILE type and functions
#include <stdio.h>
// Add definition for uint32_t
#include <stdint.h>

typedef struct starsh_particles
//! Structure for general N-body problems.
{
    size_t count;
    //!< Number of spatial points.
    int ndim;
    //!< Dimensionality of the problem.
    double *point;
    //!< Coordinates of particles.
} STARSH_particles;

void starsh_particles_free(STARSH_particles *data);

int starsh_particles_generate(STARSH_particles **data, size_t count, int ndim,
        enum STARSH_PARTICLES_PLACEMENT ptype);

int starsh_particles_generate_rand(STARSH_particles **data, size_t count,
        int ndim);
//int starsh_particles_generate_randn(STARSH_particles **data, size_t count,
//        int ndim);
int starsh_particles_generate_randgrid(STARSH_particles **data, size_t count,
        int ndim);
//int starsh_particles_generate_randngrid(STARSH_particles **data, size_t count,
//        int ndim);
int starsh_particles_generate_uniform(STARSH_particles **data, size_t count,
        int ndim);
int starsh_particles_generate_quasiuniform1(STARSH_particles **data,
        size_t count, int ndim);
int starsh_particles_generate_quasiuniform2(STARSH_particles **data,
        size_t count, int ndim);
int starsh_particles_generate_obsolete1(STARSH_particles **data, size_t count,
        int ndim);
int starsh_particles_generate_obsolete2(STARSH_particles **data, size_t count,
        int ndim);


int starsh_particles_read_from_file(STARSH_particles **data, const char *fname,
        const enum STARSH_FILE_TYPE ftype);
int starsh_particles_read_from_file_pointer(STARSH_particles **data,
        FILE *fp, const enum STARSH_FILE_TYPE ftype);
int starsh_particles_read_from_file_pointer_ascii(STARSH_particles **data,
        FILE *fp);
int starsh_particles_read_from_file_pointer_binary(STARSH_particles **data,
        FILE *fp);

int starsh_particles_write_to_file(const STARSH_particles *data,
        const char *fname, const enum STARSH_FILE_TYPE ftype);
int starsh_particles_write_to_file_pointer(const STARSH_particles *data,
        FILE *fp, const enum STARSH_FILE_TYPE ftype);
int starsh_particles_write_to_file_pointer_ascii(const STARSH_particles *data,
        FILE *fp);
int starsh_particles_write_to_file_pointer_binary(const STARSH_particles *data,
        FILE *fp);

int starsh_particles_zsort_inplace(STARSH_particles *data);
#endif // __STARSH_PARTICLES_H__

