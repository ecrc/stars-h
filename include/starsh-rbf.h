/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-spatial.h
 * @version 0.1.1
 * @author Aleksandr Mikhalev
 * @date 2018-11-06
 * */

#ifndef __STARSH_RBF_H__
#define __STARSH_RBF_H__

/*! @defgroup app-spatial Spatial statistics problem
 * @ingroup applications
 * @brief Template for spatial statistics problems.
 *
 * @ref STARSH_ssdata holds all the necessary data.
 * */

/*! @defgroup app-spatial-kernels Kernels
 * @ingroup app-spatial
 * @brief Set of kernels for spatial statistics problems
 *
 * Click on functions to view implemented equations.
 * */

// Add definitions for size_t, va_list, STARSH_kernel and STARSH_particles
#include "starsh.h"
#include "starsh-particles.h"

typedef struct starsh_mddata
//! Structure for Spatial Statistics problems.
/*! @ingroup app-spatial
 * */
{
    STARSH_particles particles;
    //!< Particles.
    //char dtype;
    ////!< Precision of each matrix element (double, single etc).
    double reg;
    //!< Noise and regularization parameter.
    int kernel;
    int numobj;
    int isreg;
    double rad;
    int mesh_points;
    int mordering;
} STARSH_mddata;

void starsh_generate_3d_virus(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int lda);
void starsh_generate_3d_virus_rhs(STARSH_int mesh_points, double *A);
void starsh_generate_3d_cube(int nrows, int ncols,
        STARSH_int *irow, STARSH_int *icol, void *row_data, void *col_data,
        void *result, int lda);
void starsh_generate_3d_virus_rhs(STARSH_int mesh_points, double *A);
void starsh_generate_3d_rbf_mesh_coordinates(STARSH_mddata **data, char *file_name, STARSH_int mesh_points, int ndim, int kernel, 
                          int numobj, int isreg, double reg, double rad, int mordering);

#endif // __STARSH_RBF__H__