/*! @copyright (c) 2017 King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 *
 * STARS-H is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file include/starsh-mpi.h
 * @version 0.1.0
 * @author Aleksandr Mikhalev
 * @date 2017-11-07
 * */

#ifndef __STARSH_MPI_H__
#define __STARSH_MPI_H__


///////////////////////////////////////////////////////////////////////////////
//                                H-FORMAT                                   //
///////////////////////////////////////////////////////////////////////////////

// Check if this is enabled in Doxygen
//! @cond (MPI)

/*! @addtogroup blrf
 * @{
 * */
// This will automatically include all entities between @{ and @} into group.

int starsh_blrf_new_from_coo_mpi(STARSH_blrf **format, STARSH_problem *problem,
        char symm, STARSH_cluster *row_cluster, STARSH_cluster *col_cluster,
        STARSH_int nblocks_far, STARSH_int *block_far,
        STARSH_int nblocks_far_local, STARSH_int *block_far_local,
        STARSH_int nblocks_near, STARSH_int *block_near,
        STARSH_int nblocks_near_local, STARSH_int *block_near_local,
        enum STARSH_BLRF_TYPE type);
int starsh_blrf_new_tlr_mpi(STARSH_blrf **format, STARSH_problem *problem,
        char symm, STARSH_cluster *row_cluster, STARSH_cluster *col_cluster);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                                H-MATRIX                                   //
///////////////////////////////////////////////////////////////////////////////

/*! @addtogroup blrm
 * @{
 * */
// This will automatically include all entities between @{ and @} into group.

int starsh_blrm_new_mpi(STARSH_blrm **matrix, STARSH_blrf *format,
        int *far_rank, Array **far_U, Array **far_V, int onfly, Array **near_D,
        void *alloc_U, void *alloc_V, void *alloc_D, char alloc_type);
void starsh_blrm_free_mpi(STARSH_blrm *matrix);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                            APPROXIMATIONS                                 //
///////////////////////////////////////////////////////////////////////////////

/*! @addtogroup approximations
 * @{
 * */
// This will automatically include all entities between @{ and @} into group.

int starsh_blrm__dsdd_mpi(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
int starsh_blrm__drsdd_mpi(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
int starsh_blrm__dqp3_mpi(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);
int starsh_blrm__dna_mpi(STARSH_blrm **matrix, STARSH_blrf *format,
        int maxrank, double tol, int onfly);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                  MATRIX-MATRIX MULTIPLICATION                             //
///////////////////////////////////////////////////////////////////////////////

/*! @addtogroup matmul
 * @{
 * */
// This will automatically include all entities between @{ and @} into group.

int starsh_blrm__dmml_mpi(STARSH_blrm *matrix, int nrhs, double alpha,
        double *A, int lda, double beta, double *B, int ldb);
int starsh_blrm__dmml_mpi_tlr(STARSH_blrm *matrix, int nrhs, double alpha,
        double *A, int lda, double beta, double *B, int ldb);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                   MEASURE APPROXIMATION ERROR                             //
///////////////////////////////////////////////////////////////////////////////

/*! @addtogroup approximation_error
 * @{
 * */
// This will automatically include all entities between @{ and @} into group.

double starsh_blrm__dfe_mpi(STARSH_blrm *matrix);

//! @}
// End of group


///////////////////////////////////////////////////////////////////////////////
//                            ITERATIVE SOLVERS                              //
///////////////////////////////////////////////////////////////////////////////

/*! @addtogroup iter
/ * @{
 * */
// This will automatically include all entities between @{ and @} into group.

int starsh_itersolvers__dcg_mpi(STARSH_blrm *matrix, int nrhs, double *B,
        int ldb, double *X, int ldx, double tol, double *work);

//! @}
// End of group

//! @endcond
// End of condition

#endif // __STARSH_MPI_H__

