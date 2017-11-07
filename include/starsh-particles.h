/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-particles.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __STARSH_PARTICLES_H__
#define __STARSH_PARTICLES_H__

/*! @defgroup app-particles N-body
 * @brief Template for particle-to-particle interactions.
 *
 * @ref STARSH_particles holds all the necessary data, which can be generated
 * in different ways by starsh_particles_generate(), read from file by
 * starsh_particles_read_from_file(), set as needed after
 * allocating memory by starsh_particles_new() or set by given coordinates by
 * starsh_particles_init().
 *
 * @ingroup applications
 * */

// Add definitions for size_t, va_list and STARSH_kernel
#include "starsh.h"
// Add definitions for FILE type and functions
#include <stdio.h>
// Add definition for uint32_t
#include <stdint.h>

typedef struct starsh_particles
//! Structure for general N-body problems.
/*! @ingroup app-particles
 * */
{
    STARSH_int count;
    //!< Number of particles.
    int ndim;
    //!< Dimensionality of space.
    double *point;
    //!< Coordinates of particles.
} STARSH_particles;

enum STARSH_PARTICLES_PLACEMENT
//! Distribution of particles for starsh_particles_generate().
/*! @ingroup app-particles
 * */
{
    STARSH_PARTICLES_RAND = 1,
    //!< Uniform random distribution in [0,1] range.
    //STARSH_PARTICLES_RANDN = 2,
    ////!< Normal random distribution with mean=0 and variance=1.
    STARSH_PARTICLES_UNIFORM = 3,
    //!< Uniform in [0,1] grid.
    STARSH_PARTICLES_RANDGRID = 4,
    //!< Grid, based on uniform in [0,1] distribution of coordinates.
    //STARSH_PARTICLES_RANDNGRID = 5,
    ////!< Grid, based on normal (0,1) ditribution of coordinates.
    STARSH_PARTICLES_QUASIUNIFORM1 = 6,
    //!< Uniform in [0,1] grid, but each particle is slightly shifted.
    STARSH_PARTICLES_QUASIUNIFORM2 = 7,
    //!< Uniform in [0,1] grid, but each grid coordinate is slightly shifted.
    STARSH_PARTICLES_OBSOLETE1 = -1,
    //!< Old version of STARSH_PARTICLES_QUASIUNIFORM1 (for compatibility).
    STARSH_PARTICLES_OBSOLETE2 = -2,
    //!< Old version of STARSH_PARTICLES_QUASIUNIFORM2 (for compatibility).
};

int starsh_particles_new(STARSH_particles **data, STARSH_int count, int ndim);
int starsh_particles_init(STARSH_particles **data, STARSH_int count, int ndim,
        double *point);
void starsh_particles_free(STARSH_particles *data);

int starsh_particles_generate(STARSH_particles **data, STARSH_int count,
        int ndim, enum STARSH_PARTICLES_PLACEMENT ptype);

int starsh_particles_generate_rand(STARSH_particles **data, STARSH_int count,
        int ndim);
int starsh_particles_generate_randgrid(STARSH_particles **data,
        STARSH_int count, int ndim);
int starsh_particles_generate_uniform(STARSH_particles **data,
        STARSH_int count, int ndim);
int starsh_particles_generate_quasiuniform1(STARSH_particles **data,
        STARSH_int count, int ndim);
int starsh_particles_generate_quasiuniform2(STARSH_particles **data,
        STARSH_int count, int ndim);
int starsh_particles_generate_obsolete1(STARSH_particles **data,
        STARSH_int count, int ndim);
int starsh_particles_generate_obsolete2(STARSH_particles **data,
        STARSH_int count, int ndim);

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

