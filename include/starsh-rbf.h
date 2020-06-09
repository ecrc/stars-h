/*! @copyright (c) 2020 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-rbf.h
 * @version 0.1.1
 * @auther Rabab Alomairy
 * @author Aleksandr Mikhalev
 * @date 2018-05-09
 * */

#ifndef __STARSH_RBF_H__
#define __STARSH_RBF_H__

/*! @defgroup mesh deformation problem
 * @ingroup applications
 * @brief Template for mesh deformation problems.
 *
 * @ref STARSH_mddata holds all the necessary data.
 * */


// Add definitions for size_t, va_list, STARSH_kernel and STARSH_particles
#include "starsh.h"
#include "starsh-particles.h"

typedef struct starsh_mddata
//! Structure for mesh deformation problems.
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
/* RBF Kernels headers */
double Gaussian(double x);
double Expon(double x);
double Maternc1(double x);
double Maternc2(double x);
double QUAD(double x);
double InvQUAD(double x);
double InvMQUAD(double x);
double TPS(double x);
double Wendland(double x);
double CTPS(double x);
double diff(double*x, double*y);


#endif // __STARSH_RBF__H__