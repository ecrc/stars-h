/*! @copyright (c) 2020 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-acoustic.h
 * @version 0.3.0
 * @auther Rabab Alomairy
 * @author Aleksandr Mikhalev
 * @date 2020-06-09
 * */

#ifndef __STARSH_ACOUSTIC_H__
#define __STARSH_ACOUSTIC_H__

/*! @defgroup acoustics scattering problem
 * @ingroup applications
 * @brief Template for acoustics scattering problems.
 *
 * @ref STARSH_acdata holds all the necessary data.
 * */


// Add definitions for size_t, va_list, STARSH_kernel and STARSH_particles
#include "starsh.h"
#include "starsh-particles.h"

typedef struct starsh_acdata
//! Structure for mesh deformation problems.
{
    char dtype;
    ////!< Precision of each matrix element (double, single etc).
    int train; // number of traingles
    int nipp; //number of quadrature points
    int mesh_points;
    int mordering;
} STARSH_acdata;


void starsh_generate_3d_acoustic(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int lda);

int starsh_generate_3d_acoustic_coordinates(STARSH_acdata **data, STARSH_int mesh_points,
                                             int ndim, int train, int nipp, int mordering, char* file_name, char* file_name_interpl);
void starsh_generate_acoustic_rhs(int nip, int ntrain, double _Complex *crhs, int m, int n, int local_nt, int nb);
void starsh_generate_acoustic_near_sca(double _Complex *rhs, int nip, int ntrian);
 

// C wrapper for Fortran
void generate_mesh_points_serials(int *nip, int *ntrain, char* file_name, int* filelength1, char* file_name_interpl, int* filelength2);
void acoustic_generate_kernel(int *nip, int *ntrian, double _Complex *zz, int *q, int *p, int *local_nt, int *nb);
void acoustic_generate_rhs(int *nip, int *ntrain, double _Complex *crhs, int *m, int *n, int *local_nt, int *nb);
void acoustic_generate_near_sca(double _Complex *rhs, int *nip, int *ntrian);

#endif // __STARSH_ACOUSTIC_H__
