/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-starpu.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __STARSH_STARPU_H__
#define __STARSH_STARPU_H__


///////////////////////////////////////////////////////////////////////////////
//                            APPROXIMATIONS                                 //
///////////////////////////////////////////////////////////////////////////////

// Check if this is enabled in Doxygen
//! @cond (STARPU)

/*! @addtogroup approximations
 * @{
 * */
// This will automatically include all entities between @{ and @} into group.

int starsh_blrm__dsdd_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
int starsh_blrm__drsdd_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
int starsh_blrm__dqp3_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
//int starsh_blrm__dna_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
//        int maxrank, double tol, int onfly);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                  LOW-RANK ROUTINES FOR DENSE                              //
///////////////////////////////////////////////////////////////////////////////

/*! @addtogroup lrdense
 * @{
 * */
// This will automatically include all entities between @{ and @} into group.

void starsh_dense_dlrsdd_starpu(void *buffers[], void *cl_arg);
void starsh_dense_dlrrsdd_starpu(void *buffers[], void *cl_arg);
void starsh_dense_dlrqp3_starpu(void *buffers[], void *cl_arg);
void starsh_dense_kernel_starpu(void *buffers[], void *cl_arg);
void starsh_dense_dgemm_starpu(void *buffers[], void *cl_arg);
void starsh_dense_fake_init_starpu(void *buffers[], void *cl_arg);

//! @}
// End of group

//! @endcond
// End of condition

#endif // __STARSH_STARPU_H__

