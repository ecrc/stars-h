/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-mpi-starpu.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __STARSH_MPI_STARPU_H__
#define __STARSH_MPI_STARPU_H__


///////////////////////////////////////////////////////////////////////////////
//                            APPROXIMATIONS                                 //
///////////////////////////////////////////////////////////////////////////////

// Check if this is enabled in Doxygen
//! @cond (STARPU && MPI)

/*! @addtogroup approximations
 * @{
 * */
// This will automatically include all entities between @{ and @} into group.

int starsh_blrm__dsdd_mpi_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
int starsh_blrm__drsdd_mpi_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
int starsh_blrm__dqp3_mpi_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
//int starsh_blrm__dna_mpi_starpu(STARSH_blrm **matrix, STARSH_blrf *format,
//        int maxrank, double tol, int onfly);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                  MATRIX-MATRIX MULTIPLICATION                             //
///////////////////////////////////////////////////////////////////////////////

/*! @addtogroup matmul
 * @{
 * */
// This will automatically include all entities between @{ and @} into group.

//int starsh_blrm__dmml_mpi_starpu(STARSH_blrm *matrix, int nrhs, double alpha,
//        double *A, int lda, double beta, double *B, int ldb);
//int starsh_blrm__dmml_mpi_starpu_tlr(STARSH_blrm *matrix, int nrhs,
//        double alpha, double *A, int lda, double beta, double *B, int ldb);

//! @}
// End of group

//! @endcond
// End of condition

#endif // __STARSH_MPI_STARPU_H__

